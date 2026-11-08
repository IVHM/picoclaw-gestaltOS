package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/providers"
)

// DelegationGuardHookName is the builtin hook name used in config
// (hooks.builtins.delegation_guard).
const DelegationGuardHookName = "delegation_guard"

// delegationGuardMaxTurns bounds the per-turn state kept in memory.
const delegationGuardMaxTurns = 256

// DelegationGuardConfig is the builtin's config block.
type DelegationGuardConfig struct {
	// MaxDirectCalls is how many non-delegating tool calls a top-level turn may
	// make itself. Later calls are denied with a reason that points at spawn.
	MaxDirectCalls int `json:"max_direct_calls"`
	// DelegateTools are always allowed and never counted.
	DelegateTools []string `json:"delegate_tools"`
	// Remind adds a short budget note before each model call of a top-level turn.
	Remind *bool `json:"remind"`
	// FinalAnswerGate sends a top-level turn back once when it tries to answer
	// after going over the direct budget without spawning any sub-agent.
	FinalAnswerGate *bool `json:"final_answer_gate"`
	// SubagentPrompt is a note added before each model call of a sub-turn, e.g.
	// "you are a sub-agent: do the task yourself". SubagentPromptFile reads it
	// from a file instead (read once, when the hook is built). Empty: no note.
	SubagentPrompt     string `json:"subagent_prompt"`
	SubagentPromptFile string `json:"subagent_prompt_file"`
}

// DelegationGuard keeps top-level turns to a small direct tool budget so that
// multi-step work goes to sub-agents. Sub-turns are not restricted.
type DelegationGuard struct {
	max      int
	delegate map[string]bool
	remind   bool
	gate     bool
	subNote  string

	mu    sync.Mutex
	turns map[string]*delegationTurn
	order []string

	// steer queues a user message on a session so the running turn continues
	// instead of finishing. Bound when the hook is mounted on an AgentLoop.
	steer func(sessionKey, agentID string, msg providers.Message) error
}

// delegationTurn is what the guard knows about one top-level turn.
type delegationTurn struct {
	attempts int  // direct calls tried, denied ones included
	spawned  bool // a delegating tool was called
	gated    bool // the final-answer gate already sent this turn back
}

func init() {
	_ = RegisterBuiltinHook(DelegationGuardHookName, newDelegationGuardFromSpec)
}

func newDelegationGuardFromSpec(_ context.Context, spec config.BuiltinHookConfig) (any, error) {
	cfg := DelegationGuardConfig{}
	if len(spec.Config) > 0 {
		if err := json.Unmarshal(spec.Config, &cfg); err != nil {
			return nil, fmt.Errorf("delegation_guard config: %w", err)
		}
	}
	if cfg.SubagentPromptFile != "" {
		data, err := os.ReadFile(cfg.SubagentPromptFile)
		if err != nil {
			return nil, fmt.Errorf("delegation_guard subagent_prompt_file: %w", err)
		}
		cfg.SubagentPrompt = string(data)
	}
	return NewDelegationGuard(cfg), nil
}

func NewDelegationGuard(cfg DelegationGuardConfig) *DelegationGuard {
	if cfg.MaxDirectCalls <= 0 {
		cfg.MaxDirectCalls = 2
	}
	if len(cfg.DelegateTools) == 0 {
		cfg.DelegateTools = []string{"spawn", "spawn_status", "subagent"}
	}
	g := &DelegationGuard{
		max:      cfg.MaxDirectCalls,
		delegate: map[string]bool{},
		remind:   cfg.Remind == nil || *cfg.Remind,
		gate:     cfg.FinalAnswerGate == nil || *cfg.FinalAnswerGate,
		subNote:  strings.TrimSpace(cfg.SubagentPrompt),
		turns:    map[string]*delegationTurn{},
	}
	for _, name := range cfg.DelegateTools {
		g.delegate[name] = true
	}
	return g
}

// bindAgentLoop gives the guard the steering queue it needs for the
// final-answer gate. MountHook calls it.
func (g *DelegationGuard) bindAgentLoop(al *AgentLoop) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.steer = al.enqueueSteeringMessage
}

func (g *DelegationGuard) topLevel(meta HookMeta) bool {
	return meta.ParentTurnID == "" && meta.TurnID != ""
}

// turn returns the state of a turn, creating it. Callers hold g.mu.
func (g *DelegationGuard) turn(turnID string) *delegationTurn {
	t, ok := g.turns[turnID]
	if !ok {
		t = &delegationTurn{}
		g.turns[turnID] = t
		g.order = append(g.order, turnID)
		if len(g.order) > delegationGuardMaxTurns {
			delete(g.turns, g.order[0])
			g.order = g.order[1:]
		}
	}
	return t
}

func (g *DelegationGuard) BeforeLLM(_ context.Context, req *LLMHookRequest) (*LLMHookRequest, HookDecision, error) {
	if req == nil {
		return req, HookDecision{Action: HookActionContinue}, nil
	}
	if req.Meta.ParentTurnID != "" {
		// A sub-turn sees the workspace's coordinator rules too; this note tells it to do its task.
		if g.subNote == "" {
			return req, HookDecision{Action: HookActionContinue}, nil
		}
		req.Messages = append(req.Messages, providers.Message{Role: "user", Content: g.subNote})
		return req, HookDecision{Action: HookActionModify}, nil
	}
	if !g.remind || !g.topLevel(req.Meta) {
		return req, HookDecision{Action: HookActionContinue}, nil
	}
	g.mu.Lock()
	t, ok := g.turns[req.Meta.TurnID]
	used, gated := 0, false
	if ok {
		used, gated = t.attempts, t.gated
	}
	g.mu.Unlock()
	if gated {
		// The final-answer gate's note is the last word for this turn.
		return req, HookDecision{Action: HookActionContinue}, nil
	}
	left := g.max - used
	var note string
	if left > 0 {
		note = fmt.Sprintf("[delegation guard] Direct tool budget for this turn: %d of %d left. "+
			"If the task needs more than %d tool steps, plan it now and hand each multi-step piece "+
			"to the spawn tool; keep this turn for coordinating and reporting.", left, g.max, g.max)
	} else {
		note = fmt.Sprintf("[delegation guard] Direct tool budget for this turn is spent (%d of %d). "+
			"Further direct tool calls will be denied. Delegate the remaining work with the spawn tool, "+
			"or answer from what you already have.", g.max, g.max)
	}
	req.Messages = append(req.Messages, providers.Message{Role: "user", Content: note})
	return req, HookDecision{Action: HookActionModify}, nil
}

// AfterLLM is the final-answer gate. When a top-level turn answers after
// trying more direct calls than its budget and spawning nothing, it queues one
// steering message on the session. The pipeline then continues the turn with
// that message instead of finishing it. It does this at most once per turn,
// so the second answer always goes through.
func (g *DelegationGuard) AfterLLM(_ context.Context, resp *LLMHookResponse) (*LLMHookResponse, HookDecision, error) {
	if !g.gate || resp == nil || resp.Response == nil || !g.topLevel(resp.Meta) ||
		len(resp.Response.ToolCalls) > 0 || strings.TrimSpace(resp.Response.Content) == "" {
		return resp, HookDecision{Action: HookActionContinue}, nil
	}
	g.mu.Lock()
	t, ok := g.turns[resp.Meta.TurnID]
	steer := g.steer
	if !ok || steer == nil || t.gated || t.spawned || t.attempts <= g.max {
		g.mu.Unlock()
		return resp, HookDecision{Action: HookActionContinue}, nil
	}
	t.gated = true
	attempts := t.attempts
	g.mu.Unlock()

	note := fmt.Sprintf("[delegation guard] Not done yet: this turn tried %d direct tool calls (budget %d) "+
		"and delegated nothing. Hand the remaining work to sub-agents with the spawn tool, then report. "+
		"If delegating makes no sense here, reply again and say briefly why; that reply will be sent.",
		attempts, g.max)
	// If queueing fails the answer simply goes through; the turn is marked gated either way, so
	// the gate can never loop.
	_ = steer(resp.Meta.SessionKey, resp.Meta.AgentID, providers.Message{Role: "user", Content: note})
	return resp, HookDecision{Action: HookActionContinue}, nil
}

func (g *DelegationGuard) BeforeTool(
	_ context.Context,
	call *ToolCallHookRequest,
) (*ToolCallHookRequest, HookDecision, error) {
	if call == nil || !g.topLevel(call.Meta) {
		return call, HookDecision{Action: HookActionContinue}, nil
	}
	g.mu.Lock()
	t := g.turn(call.Meta.TurnID)
	if g.delegate[call.Tool] {
		t.spawned = true
		g.mu.Unlock()
		return call, HookDecision{Action: HookActionContinue}, nil
	}
	t.attempts++
	over := t.attempts > g.max
	g.mu.Unlock()
	if over {
		return call, HookDecision{
			Action: HookActionDenyTool,
			Reason: fmt.Sprintf("this turn may make only %d direct tool calls; this is a multi-step task, "+
				"so delegate it with the spawn tool (give the sub-agent the full task and what you found so far)", g.max),
		}, nil
	}
	return call, HookDecision{Action: HookActionContinue}, nil
}

func (g *DelegationGuard) AfterTool(
	_ context.Context,
	result *ToolResultHookResponse,
) (*ToolResultHookResponse, HookDecision, error) {
	return result, HookDecision{Action: HookActionContinue}, nil
}
