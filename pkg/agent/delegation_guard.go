package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/providers"
)

// DelegationGuardHookName is the builtin hook name used in config
// (hooks.builtins.delegation_guard).
const DelegationGuardHookName = "delegation_guard"

// delegationGuardMaxTurns bounds the per-turn counters kept in memory.
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
}

// DelegationGuard keeps top-level turns to a small direct tool budget so that
// multi-step work goes to sub-agents. Sub-turns are not restricted.
type DelegationGuard struct {
	max      int
	delegate map[string]bool
	remind   bool

	mu    sync.Mutex
	used  map[string]int
	order []string
}

func init() {
	_ = RegisterBuiltinHook(DelegationGuardHookName, func(_ context.Context, spec config.BuiltinHookConfig) (any, error) {
		cfg := DelegationGuardConfig{}
		if len(spec.Config) > 0 {
			if err := json.Unmarshal(spec.Config, &cfg); err != nil {
				return nil, fmt.Errorf("delegation_guard config: %w", err)
			}
		}
		return NewDelegationGuard(cfg), nil
	})
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
		used:     map[string]int{},
	}
	for _, name := range cfg.DelegateTools {
		g.delegate[name] = true
	}
	return g
}

func (g *DelegationGuard) topLevel(meta HookMeta) bool {
	return meta.ParentTurnID == "" && meta.TurnID != ""
}

func (g *DelegationGuard) count(turnID string) int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.used[turnID]
}

// take records one direct call and returns the new total.
func (g *DelegationGuard) take(turnID string) int {
	g.mu.Lock()
	defer g.mu.Unlock()
	if _, ok := g.used[turnID]; !ok {
		g.order = append(g.order, turnID)
		if len(g.order) > delegationGuardMaxTurns {
			delete(g.used, g.order[0])
			g.order = g.order[1:]
		}
	}
	g.used[turnID]++
	return g.used[turnID]
}

func (g *DelegationGuard) BeforeLLM(_ context.Context, req *LLMHookRequest) (*LLMHookRequest, HookDecision, error) {
	if !g.remind || req == nil || !g.topLevel(req.Meta) {
		return req, HookDecision{Action: HookActionContinue}, nil
	}
	left := g.max - g.count(req.Meta.TurnID)
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

func (g *DelegationGuard) AfterLLM(_ context.Context, resp *LLMHookResponse) (*LLMHookResponse, HookDecision, error) {
	return resp, HookDecision{Action: HookActionContinue}, nil
}

func (g *DelegationGuard) BeforeTool(_ context.Context, call *ToolCallHookRequest) (*ToolCallHookRequest, HookDecision, error) {
	if call == nil || !g.topLevel(call.Meta) || g.delegate[call.Tool] {
		return call, HookDecision{Action: HookActionContinue}, nil
	}
	if n := g.take(call.Meta.TurnID); n > g.max {
		return call, HookDecision{
			Action: HookActionDenyTool,
			Reason: fmt.Sprintf("this turn may make only %d direct tool calls; this is a multi-step task, "+
				"so delegate it with the spawn tool (give the sub-agent the full task and what you found so far)", g.max),
		}, nil
	}
	return call, HookDecision{Action: HookActionContinue}, nil
}

func (g *DelegationGuard) AfterTool(_ context.Context, result *ToolResultHookResponse) (*ToolResultHookResponse, HookDecision, error) {
	return result, HookDecision{Action: HookActionContinue}, nil
}
