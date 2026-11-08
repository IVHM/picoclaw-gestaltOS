package tools

import (
	"context"
	"strings"
	"testing"
	"time"
)

type fakePrepareTool struct {
	got    map[string]any
	result *ToolResult
}

func (f *fakePrepareTool) Name() string               { return "mcp_bridge_delegate_prepare" }
func (f *fakePrepareTool) Description() string        { return "prepare" }
func (f *fakePrepareTool) Parameters() map[string]any { return map[string]any{"type": "object"} }
func (f *fakePrepareTool) Execute(_ context.Context, args map[string]any) *ToolResult {
	f.got = args
	return f.result
}

type fakeSpawner struct{ cfgs chan SubTurnConfig }

func (f *fakeSpawner) SpawnSubTurn(_ context.Context, cfg SubTurnConfig) (*ToolResult, error) {
	f.cfgs <- cfg
	return &ToolResult{ForLLM: "worker done"}, nil
}

func TestDelegateTaskSpawnsTheWorkerWithExactlyTheBuiltPrompt(t *testing.T) {
	prompt := "Task TI-7f3a: audit.\nbundle_id CB-0123456789 (pass unchanged)"
	prep := &fakePrepareTool{result: &ToolResult{ForLLM: `{"prompt":` + quote(prompt) +
		`,"ticket_id":"TI-7f3a","bundle_id":"CB-0123456789","procedure_id":"proc.qa.x"}`}}
	sp := &fakeSpawner{cfgs: make(chan SubTurnConfig, 1)}
	tool := NewDelegateTaskTool(
		nil,
		func(name string) (Tool, bool) { return prep, name == prep.Name() },
		prep.Name(),
		"worker",
	)
	tool.SetSpawner(sp)

	done := make(chan *ToolResult, 1)
	res := tool.ExecuteAsync(context.Background(), map[string]any{
		"problem":          "Check the heartbeat files.",
		"resources":        []any{"/root/.picoclaw/workspace/HEARTBEAT.md"},
		"success_criteria": []any{"both files read", "sizes reported"},
	}, func(_ context.Context, r *ToolResult) { done <- r })
	if res.IsError || !res.Async || !strings.Contains(res.ForLLM, "TI-7f3a") ||
		!strings.Contains(res.ForLLM, `agent "worker"`) {
		t.Fatalf("result = %+v", res)
	}
	if prep.got["problem"] != "Check the heartbeat files." || len(prep.got["success_criteria"].([]string)) != 2 ||
		prep.got["resources"].([]string)[0] != "/root/.picoclaw/workspace/HEARTBEAT.md" {
		t.Fatalf("prepare args = %#v", prep.got)
	}
	select {
	case cfg := <-sp.cfgs:
		if cfg.SystemPrompt != prompt || cfg.TargetAgentID != "worker" || cfg.Label != "TI-7f3a" || !cfg.Async {
			t.Fatalf("spawn cfg = %+v", cfg)
		}
	case <-time.After(time.Second):
		t.Fatal("worker not spawned")
	}
	select {
	case r := <-done:
		if r.ForLLM != "worker done" {
			t.Fatalf("callback result = %+v", r)
		}
	case <-time.After(time.Second):
		t.Fatal("no callback")
	}
}

func TestDelegateTaskErrors(t *testing.T) {
	sp := &fakeSpawner{cfgs: make(chan SubTurnConfig, 1)}
	cases := map[string]struct {
		prep *fakePrepareTool
		args map[string]any
		want string
	}{
		"no problem":      {nil, map[string]any{"success_criteria": []any{"x"}}, "problem is required"},
		"no criteria":     {nil, map[string]any{"problem": "p"}, "success_criteria"},
		"prepare missing": {nil, map[string]any{"problem": "p", "success_criteria": "x"}, "not available"},
		"prepare failed": {
			&fakePrepareTool{result: ErrorResult("ticket engine down")},
			map[string]any{"problem": "p", "success_criteria": "x"},
			"ticket engine down",
		},
		"no prompt in json": {
			&fakePrepareTool{result: &ToolResult{ForLLM: `{"ticket_id":"TI-1"}`}},
			map[string]any{"problem": "p", "success_criteria": "x"},
			"no prompt",
		},
	}
	for name, c := range cases {
		lookup := func(string) (Tool, bool) { return nil, false }
		if c.prep != nil {
			p := c.prep
			lookup = func(string) (Tool, bool) { return p, true }
		}
		tool := NewDelegateTaskTool(nil, lookup, "mcp_bridge_delegate_prepare", "worker")
		tool.SetSpawner(sp)
		res := tool.Execute(context.Background(), c.args)
		if !res.IsError || !strings.Contains(res.ForLLM, c.want) {
			t.Errorf("%s: result = %+v, want error containing %q", name, res, c.want)
		}
	}
	if len(sp.cfgs) != 0 {
		t.Fatal("no worker may start when preparing fails")
	}
}

func quote(s string) string {
	return `"` + strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`).Replace(s) + `"`
}
