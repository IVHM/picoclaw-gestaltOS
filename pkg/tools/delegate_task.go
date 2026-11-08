package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// DelegateTaskTool hands a task to a worker sub-agent whose prompt is built
// elsewhere. The caller gives the problem, the starting resources and the
// success criteria; a configured "prepare" tool (usually an MCP tool) turns
// that brief into a prompt (for example by filing a ticket and running a
// context builder) and returns it as JSON; the tool then spawns the
// configured agent with exactly that prompt, in the background.
//
// The prepare tool's result must be JSON with at least "prompt"; "ticket_id",
// "bundle_id" and "procedure_id" are reported back when present.
type DelegateTaskTool struct {
	spawner      SubTurnSpawner
	lookup       func(name string) (Tool, bool)
	prepareTool  string
	agentID      string
	defaultModel string
	maxTokens    int
	temperature  float64
}

// Compile-time check: DelegateTaskTool implements AsyncExecutor.
var _ AsyncExecutor = (*DelegateTaskTool)(nil)

// NewDelegateTaskTool builds the tool. lookup finds the prepare tool by its
// registered name at call time, so MCP tools registered later work.
func NewDelegateTaskTool(
	manager *SubagentManager,
	lookup func(name string) (Tool, bool),
	prepareTool, agentID string,
) *DelegateTaskTool {
	t := &DelegateTaskTool{lookup: lookup, prepareTool: prepareTool, agentID: agentID}
	if manager != nil {
		t.defaultModel = manager.defaultModel
		t.maxTokens = manager.maxTokens
		t.temperature = manager.temperature
	}
	return t
}

// SetSpawner sets the SubTurnSpawner used to start the worker.
func (t *DelegateTaskTool) SetSpawner(spawner SubTurnSpawner) {
	t.spawner = spawner
}

func (t *DelegateTaskTool) Name() string {
	return "delegate_task"
}

func (t *DelegateTaskTool) Description() string {
	return "Hand a task to a worker sub-agent. Give only the problem, the starting resources " +
		"(paths, links, ids) and the success criteria; its prompt is built for you. " +
		"It runs in the background and its result arrives when done."
}

func (t *DelegateTaskTool) Parameters() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"problem": map[string]any{
				"type":        "string",
				"description": "What needs to be done and why, in a few sentences",
			},
			"resources": map[string]any{
				"type":        "array",
				"items":       map[string]any{"type": "string"},
				"description": "Starting points: file paths, URLs, ticket or document ids",
			},
			"success_criteria": map[string]any{
				"type":        "array",
				"items":       map[string]any{"type": "string"},
				"description": "Checkable statements that are true when the task is done",
			},
			"label": map[string]any{
				"type":        "string",
				"description": "Optional short name for the worker",
			},
		},
		"required": []string{"problem", "success_criteria"},
	}
}

func (t *DelegateTaskTool) Execute(ctx context.Context, args map[string]any) *ToolResult {
	return t.ExecuteAsync(ctx, args, nil)
}

func (t *DelegateTaskTool) ExecuteAsync(ctx context.Context, args map[string]any, cb AsyncCallback) *ToolResult {
	problem, _ := args["problem"].(string)
	if strings.TrimSpace(problem) == "" {
		return ErrorResult("problem is required")
	}
	criteria := stringList(args["success_criteria"])
	if len(criteria) == 0 {
		return ErrorResult("success_criteria needs at least one item")
	}
	label, _ := args["label"].(string)
	if t.spawner == nil {
		return ErrorResult("delegate_task: sub-agents are not configured")
	}
	prep, ok := t.lookup(t.prepareTool)
	if !ok || prep == nil {
		return ErrorResult(
			fmt.Sprintf("delegate_task: prepare tool %q is not available (is its MCP server up?)", t.prepareTool),
		)
	}
	res := prep.Execute(ctx, map[string]any{
		"problem":          problem,
		"resources":        stringList(args["resources"]),
		"success_criteria": criteria,
		"label":            label,
	})
	if res == nil || res.IsError {
		msg := "no result"
		if res != nil {
			msg = res.ForLLM
		}
		return ErrorResult("delegate_task: preparing the task failed: " + msg)
	}
	var built struct {
		Prompt      string `json:"prompt"`
		TicketID    string `json:"ticket_id"`
		BundleID    string `json:"bundle_id"`
		ProcedureID string `json:"procedure_id"`
	}
	if err := json.Unmarshal(
		[]byte(strings.TrimSpace(res.ForLLM)),
		&built,
	); err != nil ||
		strings.TrimSpace(built.Prompt) == "" {
		return ErrorResult("delegate_task: the prepare tool returned no prompt (expected JSON with \"prompt\"; " +
			"a large result may have been saved as an artifact: raise tools.mcp.max_inline_text_chars)")
	}
	if label == "" {
		label = built.TicketID
	}

	go func() {
		result, err := t.spawner.SpawnSubTurn(ctx, SubTurnConfig{
			Model:         t.defaultModel,
			SystemPrompt:  built.Prompt, // passed to the worker exactly as built
			MaxTokens:     t.maxTokens,
			Temperature:   t.temperature,
			Async:         true,
			Critical:      true,
			TargetAgentID: t.agentID,
			Label:         label,
		})
		if err != nil {
			result = ErrorResult(fmt.Sprintf("delegate_task: starting the worker failed: %v", err)).WithError(err)
		}
		if cb != nil {
			cb(ctx, result)
		}
	}()

	parts := []string{}
	for _, kv := range [][2]string{{"ticket", built.TicketID}, {"bundle", built.BundleID}, {"procedure", built.ProcedureID}} {
		if kv[1] != "" {
			parts = append(parts, kv[0]+" "+kv[1])
		}
	}
	who := "a worker"
	if t.agentID != "" {
		who = fmt.Sprintf("agent %q", t.agentID)
	}
	return AsyncResult(fmt.Sprintf("Delegated to %s as %s. It runs in the background; its result arrives when done.",
		who, strings.Join(parts, ", ")))
}

// stringList reads a JSON array of strings (or one string) from tool args.
func stringList(v any) []string {
	switch x := v.(type) {
	case []string:
		return x
	case string:
		if strings.TrimSpace(x) == "" {
			return nil
		}
		return []string{x}
	case []any:
		out := make([]string, 0, len(x))
		for _, item := range x {
			if s, ok := item.(string); ok && strings.TrimSpace(s) != "" {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}
