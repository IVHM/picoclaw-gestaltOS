package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/providers"
)

func TestDelegationGuardDeniesAfterBudget(t *testing.T) {
	g := NewDelegationGuard(DelegationGuardConfig{MaxDirectCalls: 2})
	ctx := context.Background()
	call := func(turn, parent, tool string) HookAction {
		_, d, _ := g.BeforeTool(ctx, &ToolCallHookRequest{
			Meta: HookMeta{TurnID: turn, ParentTurnID: parent},
			Tool: tool,
		})
		return d.normalizedAction()
	}

	if call("t1", "", "read_file") != HookActionContinue || call("t1", "", "list_dir") != HookActionContinue {
		t.Fatal("first two direct calls should pass")
	}
	if got := call("t1", "", "read_file"); got != HookActionDenyTool {
		t.Fatalf("third direct call: got %q, want deny_tool", got)
	}
	if call("t1", "", "spawn") != HookActionContinue {
		t.Fatal("spawn should always pass")
	}
	for i := 0; i < 5; i++ {
		if call("sub", "t1", "read_file") != HookActionContinue {
			t.Fatal("sub-turns should not be limited")
		}
	}
	if call("t2", "", "read_file") != HookActionContinue {
		t.Fatal("a new turn gets a fresh budget")
	}
}

func TestDelegationGuardReminder(t *testing.T) {
	g := NewDelegationGuard(DelegationGuardConfig{MaxDirectCalls: 1})
	ctx := context.Background()
	req, d, _ := g.BeforeLLM(ctx, &LLMHookRequest{Meta: HookMeta{TurnID: "t1"}})
	if d.Action != HookActionModify || !strings.Contains(req.Messages[0].Content, "1 of 1 left") {
		t.Fatalf("expected budget note, got %+v", req.Messages)
	}
	if _, first, err := g.BeforeTool(
		ctx,
		&ToolCallHookRequest{Meta: HookMeta{TurnID: "t1"}, Tool: "read_file"},
	); err != nil ||
		first.Action != HookActionContinue {
		t.Fatalf("first call should pass, got %+v, %v", first, err)
	}
	req, _, _ = g.BeforeLLM(ctx, &LLMHookRequest{Meta: HookMeta{TurnID: "t1"}})
	if !strings.Contains(req.Messages[0].Content, "spent") {
		t.Fatalf("expected spent note, got %+v", req.Messages)
	}
	req, d, _ = g.BeforeLLM(ctx, &LLMHookRequest{Meta: HookMeta{TurnID: "s1", ParentTurnID: "t1"}})
	if d.Action != HookActionContinue || len(req.Messages) != 0 {
		t.Fatal("sub-turns get no note")
	}
}

func TestDelegationGuardFinalAnswerGateSendsTurnBackOnce(t *testing.T) {
	g := NewDelegationGuard(DelegationGuardConfig{MaxDirectCalls: 2})
	var steered []string
	g.steer = func(sessionKey, _ string, msg providers.Message) error {
		steered = append(steered, sessionKey+": "+msg.Content)
		return nil
	}
	meta := HookMeta{TurnID: "t1", SessionKey: "s1"}
	for i := 0; i < 3; i++ {
		tryTool(g, &ToolCallHookRequest{Meta: meta, Tool: "read_file"})
	}
	answer := &LLMHookResponse{Meta: meta, Response: &providers.LLMResponse{Content: "done"}}

	mustContinueAfterLLM(t, g, answer)
	if len(steered) != 1 || !strings.HasPrefix(steered[0], "s1: [delegation guard] Not done yet") {
		t.Fatalf("first answer should be sent back once, steered = %q", steered)
	}
	mustContinueAfterLLM(t, g, answer)
	if len(steered) != 1 {
		t.Fatalf("second answer must go through, steered = %q", steered)
	}
}

func TestDelegationGuardFinalAnswerGateLetsGoodTurnsThrough(t *testing.T) {
	answer := func(meta HookMeta) *LLMHookResponse {
		return &LLMHookResponse{Meta: meta, Response: &providers.LLMResponse{Content: "done"}}
	}
	cases := map[string]func(g *DelegationGuard){
		"within budget": func(g *DelegationGuard) {
			for i := 0; i < 2; i++ {
				tryTool(g, &ToolCallHookRequest{Meta: HookMeta{TurnID: "t1"}, Tool: "read_file"})
			}
			mustContinueAfterLLM(t, g, answer(HookMeta{TurnID: "t1"}))
		},
		"over budget but spawned": func(g *DelegationGuard) {
			for _, tool := range []string{"read_file", "read_file", "read_file", "spawn"} {
				tryTool(g, &ToolCallHookRequest{Meta: HookMeta{TurnID: "t1"}, Tool: tool})
			}
			mustContinueAfterLLM(t, g, answer(HookMeta{TurnID: "t1"}))
		},
		"sub-turn": func(g *DelegationGuard) {
			meta := HookMeta{TurnID: "s1", ParentTurnID: "t1"}
			for i := 0; i < 5; i++ {
				tryTool(g, &ToolCallHookRequest{Meta: meta, Tool: "read_file"})
			}
			mustContinueAfterLLM(t, g, answer(meta))
		},
		"tool calls are not an answer": func(g *DelegationGuard) {
			for i := 0; i < 3; i++ {
				tryTool(g, &ToolCallHookRequest{Meta: HookMeta{TurnID: "t1"}, Tool: "read_file"})
			}
			mustContinueAfterLLM(t, g, &LLMHookResponse{Meta: HookMeta{TurnID: "t1"}, Response: &providers.LLMResponse{
				ToolCalls: []providers.ToolCall{{ID: "c1", Name: "spawn"}},
			}})
		},
	}
	for name, run := range cases {
		g := NewDelegationGuard(DelegationGuardConfig{MaxDirectCalls: 2})
		steered := 0
		g.steer = func(string, string, providers.Message) error { steered++; return nil }
		run(g)
		if steered != 0 {
			t.Errorf("%s: gate fired %d times, want 0", name, steered)
		}
	}
	off := false
	g := NewDelegationGuard(DelegationGuardConfig{MaxDirectCalls: 2, FinalAnswerGate: &off})
	steered := 0
	g.steer = func(string, string, providers.Message) error { steered++; return nil }
	for i := 0; i < 3; i++ {
		tryTool(g, &ToolCallHookRequest{Meta: HookMeta{TurnID: "t1"}, Tool: "read_file"})
	}
	mustContinueAfterLLM(t, g, answer(HookMeta{TurnID: "t1"}))
	if steered != 0 {
		t.Errorf("final_answer_gate=false: gate fired")
	}
}

// mustContinueAfterLLM runs AfterLLM; the gate never changes the answer itself.
func mustContinueAfterLLM(t *testing.T, g *DelegationGuard, resp *LLMHookResponse) {
	t.Helper()
	if _, d, err := g.AfterLLM(context.Background(), resp); err != nil || d.Action != HookActionContinue {
		t.Fatalf("AfterLLM = %+v, %v; want continue", d, err)
	}
}

// tryTool runs BeforeTool and reports whether the call was allowed.
func tryTool(g *DelegationGuard, call *ToolCallHookRequest) bool {
	_, d, err := g.BeforeTool(context.Background(), call)
	return err == nil && d.normalizedAction() == HookActionContinue
}

// gateScriptProvider asks for three direct calls, then answers twice.
type gateScriptProvider struct {
	calls    int
	lastSeen []providers.Message
}

func (p *gateScriptProvider) Chat(
	_ context.Context,
	messages []providers.Message,
	_ []providers.ToolDefinition,
	_ string,
	_ map[string]any,
) (*providers.LLMResponse, error) {
	p.calls++
	p.lastSeen = messages
	switch p.calls {
	case 1:
		var calls []providers.ToolCall
		for i := 0; i < 3; i++ {
			calls = append(calls, providers.ToolCall{
				ID: fmt.Sprintf("call_%d", i), Type: "function", Name: "tool_limit_test_tool",
				Arguments: map[string]any{"value": "x"},
			})
		}
		return &providers.LLMResponse{ToolCalls: calls}, nil
	case 2:
		return &providers.LLMResponse{Content: "first answer"}, nil
	default:
		return &providers.LLMResponse{Content: "second answer: delegating makes no sense for this"}, nil
	}
}

func (p *gateScriptProvider) GetDefaultModel() string { return "gate-script-model" }

func TestDelegationGuardFinalAnswerGateInTheLoop(t *testing.T) {
	cfg := &config.Config{Agents: config.AgentsConfig{Defaults: config.AgentDefaults{
		Workspace: t.TempDir(), ModelName: "test-model", MaxTokens: 4096, MaxToolIterations: 10,
	}}}
	provider := &gateScriptProvider{}
	al := NewAgentLoop(cfg, bus.NewMessageBus(), provider)
	al.RegisterTool(&toolLimitTestTool{})
	if err := al.MountHook(
		NamedHook(DelegationGuardHookName, NewDelegationGuard(DelegationGuardConfig{MaxDirectCalls: 2})),
	); err != nil {
		t.Fatal(err)
	}

	response, err := al.ProcessDirectWithChannel(context.Background(), "do it", "gate", "test", "chat1")
	if err != nil {
		t.Fatal(err)
	}
	if provider.calls != 3 {
		t.Fatalf("model calls = %d, want 3 (tools, refused answer, accepted answer)", provider.calls)
	}
	if !strings.HasPrefix(response, "second answer") {
		t.Fatalf("response = %q, want the second answer", response)
	}
	last := provider.lastSeen[len(provider.lastSeen)-1]
	if last.Role != "user" || !strings.Contains(last.Content, "Not done yet") {
		t.Fatalf("third call should end with the gate note, got %s: %q", last.Role, last.Content)
	}
}

func TestDelegationGuardSubagentPrompt(t *testing.T) {
	ctx := context.Background()
	g := NewDelegationGuard(DelegationGuardConfig{SubagentPrompt: "  You are a sub-agent: do the task yourself.  "})
	req, d, _ := g.BeforeLLM(ctx, &LLMHookRequest{Meta: HookMeta{TurnID: "c1", ParentTurnID: "t1"}})
	if d.Action != HookActionModify || len(req.Messages) != 1 ||
		req.Messages[0].Content != "You are a sub-agent: do the task yourself." || req.Messages[0].Role != "user" {
		t.Fatalf("sub-turn note: %+v %+v", d, req.Messages)
	}
	// The main turn still gets the budget note, not the sub-agent one.
	req, _, _ = g.BeforeLLM(ctx, &LLMHookRequest{Meta: HookMeta{TurnID: "t1"}})
	if len(req.Messages) != 1 || !strings.Contains(req.Messages[0].Content, "Direct tool budget") {
		t.Fatalf("main turn: %+v", req.Messages)
	}
	// Without a prompt, sub-turns are left alone.
	g = NewDelegationGuard(DelegationGuardConfig{})
	req, d, _ = g.BeforeLLM(ctx, &LLMHookRequest{Meta: HookMeta{TurnID: "c1", ParentTurnID: "t1"}})
	if d.Action != HookActionContinue || len(req.Messages) != 0 {
		t.Fatalf("no prompt: %+v %+v", d, req.Messages)
	}
}

func TestDelegationGuardSubagentPromptFile(t *testing.T) {
	path := t.TempDir() + "/SUBAGENT.md"
	if err := os.WriteFile(path, []byte("Do it yourself.\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(map[string]any{"subagent_prompt_file": path})
	hook, err := newDelegationGuardFromSpec(context.Background(), config.BuiltinHookConfig{Config: raw})
	if err != nil {
		t.Fatal(err)
	}
	if g := hook.(*DelegationGuard); g.subNote != "Do it yourself." {
		t.Fatalf("subNote = %q", g.subNote)
	}
	raw, _ = json.Marshal(map[string]any{"subagent_prompt_file": path + ".missing"})
	if _, err := newDelegationGuardFromSpec(context.Background(), config.BuiltinHookConfig{Config: raw}); err == nil {
		t.Fatal("a missing prompt file should fail the hook config")
	}
}
