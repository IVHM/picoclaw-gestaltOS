package agent

import (
	"context"
	"strings"
	"testing"
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
	_, _, _ = g.BeforeTool(ctx, &ToolCallHookRequest{Meta: HookMeta{TurnID: "t1"}, Tool: "read_file"})
	req, _, _ = g.BeforeLLM(ctx, &LLMHookRequest{Meta: HookMeta{TurnID: "t1"}})
	if !strings.Contains(req.Messages[0].Content, "spent") {
		t.Fatalf("expected spent note, got %+v", req.Messages)
	}
	req, d, _ = g.BeforeLLM(ctx, &LLMHookRequest{Meta: HookMeta{TurnID: "s1", ParentTurnID: "t1"}})
	if d.Action != HookActionContinue || len(req.Messages) != 0 {
		t.Fatal("sub-turns get no note")
	}
}
