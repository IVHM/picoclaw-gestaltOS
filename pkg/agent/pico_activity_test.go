package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/providers"
	"github.com/sipeed/picoclaw/pkg/tools"
	"github.com/sipeed/picoclaw/pkg/utils"
)

// assertPicoToolResultPairsWithCall checks that result is a tool_result
// message whose tool_call_id and turn_id match the earlier tool_calls message.
func assertPicoToolResultPairsWithCall(t *testing.T, callMsg, resultMsg bus.OutboundMessage, wantContent string) {
	t.Helper()

	if got := callMsg.Context.Raw[metadataKeyMessageKind]; got != messageKindToolCalls {
		t.Fatalf("call message kind = %q, want %q", got, messageKindToolCalls)
	}
	var calls []utils.VisibleToolCall
	if err := json.Unmarshal([]byte(callMsg.Context.Raw[metadataKeyToolCalls]), &calls); err != nil || len(calls) == 0 {
		t.Fatalf("tool_calls raw = %q, err = %v", callMsg.Context.Raw[metadataKeyToolCalls], err)
	}
	if calls[0].ID == "" {
		t.Fatal("tool_calls entry has no id")
	}
	callTurnID := callMsg.Context.Raw[metadataKeyTurnID]
	if callTurnID == "" {
		t.Fatal("tool_calls message has no turn_id")
	}

	if got := resultMsg.Context.Raw[metadataKeyMessageKind]; got != messageKindToolResult {
		t.Fatalf("result message kind = %q, want %q (msg %+v)", got, messageKindToolResult, resultMsg)
	}
	var result PicoToolResult
	if err := json.Unmarshal([]byte(resultMsg.Context.Raw[metadataKeyToolResult]), &result); err != nil {
		t.Fatalf("tool_result raw = %q, err = %v", resultMsg.Context.Raw[metadataKeyToolResult], err)
	}
	if result.ToolCallID != calls[0].ID {
		t.Fatalf("tool_result tool_call_id = %q, want %q", result.ToolCallID, calls[0].ID)
	}
	if result.TurnID != callTurnID {
		t.Fatalf("tool_result turn_id = %q, want %q", result.TurnID, callTurnID)
	}
	if result.Status != ToolResultStatusOK || result.IsError {
		t.Fatalf("tool_result status = %q is_error = %v, want ok", result.Status, result.IsError)
	}
	if result.Tool != "tool_limit_test_tool" {
		t.Fatalf("tool_result tool = %q", result.Tool)
	}
	if result.Content != wantContent || result.ContentLen != len([]rune(wantContent)) || result.Truncated {
		t.Fatalf("tool_result content = %q len %d truncated %v", result.Content, result.ContentLen, result.Truncated)
	}
}

func newPicoActivityTestTurn(
	t *testing.T,
	channel string,
	allowInterim bool,
) (*AgentLoop, *turnState, *bus.MessageBus) {
	t.Helper()
	al, _, msgBus, _, cleanup := newTestAgentLoop(t)
	t.Cleanup(cleanup)
	ts := &turnState{
		ctx:            context.Background(),
		turnID:         "turn-1",
		agentID:        "main",
		channel:        channel,
		chatID:         "pico:session-1",
		sessionKey:     "agent:main:pico:session-1",
		opts:           processOptions{AllowInterimPicoPublish: allowInterim},
		pendingResults: make(chan *tools.ToolResult, 4),
		concurrencySem: make(chan struct{}, testMaxConcurrentSubTurns),
		session:        &ephemeralSessionStore{},
		agent:          al.registry.GetDefaultAgent(),
	}
	return al, ts, msgBus
}

func readOutbound(t *testing.T, msgBus *bus.MessageBus) bus.OutboundMessage {
	t.Helper()
	select {
	case msg := <-msgBus.OutboundChan():
		return msg
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for outbound message")
	}
	return bus.OutboundMessage{}
}

func expectNoOutbound(t *testing.T, msgBus *bus.MessageBus) {
	t.Helper()
	select {
	case msg := <-msgBus.OutboundChan():
		t.Fatalf("unexpected outbound message: %+v", msg)
	case <-time.After(100 * time.Millisecond):
	}
}

func TestPublishPicoToolResult_ClipsContentAndMarksErrors(t *testing.T) {
	al, ts, msgBus := newPicoActivityTestTurn(t, "pico", true)

	long := strings.Repeat("é", picoToolResultMaxContentRunes+50)
	al.publishPicoToolResult(
		context.Background(),
		ts,
		"call_9",
		"exec",
		ToolResultStatusError,
		long,
		1500*time.Millisecond,
		false,
	)

	msg := readOutbound(t, msgBus)
	if msg.Context.Raw[metadataKeyMessageKind] != messageKindToolResult {
		t.Fatalf("kind = %q", msg.Context.Raw[metadataKeyMessageKind])
	}
	var got PicoToolResult
	if err := json.Unmarshal([]byte(msg.Context.Raw[metadataKeyToolResult]), &got); err != nil {
		t.Fatal(err)
	}
	if got.ToolCallID != "call_9" || got.Tool != "exec" || !got.IsError || got.Status != ToolResultStatusError {
		t.Fatalf("unexpected result: %+v", got)
	}
	if !got.Truncated || got.ContentLen != picoToolResultMaxContentRunes+50 ||
		len([]rune(got.Content)) != picoToolResultMaxContentRunes {
		t.Fatalf("clip: truncated=%v content_len=%d runes=%d", got.Truncated, got.ContentLen, len([]rune(got.Content)))
	}
	if got.DurationMs != 1500 || got.TurnID != "turn-1" {
		t.Fatalf("duration_ms=%d turn_id=%q", got.DurationMs, got.TurnID)
	}
}

func TestPublishPicoToolResult_OnlyForPicoTurnsThatShowCalls(t *testing.T) {
	al, ts, msgBus := newPicoActivityTestTurn(t, "telegram", true)
	al.publishPicoToolResult(context.Background(), ts, "call_1", "exec", ToolResultStatusOK, "x", 0, false)
	expectNoOutbound(t, msgBus)

	al, ts, msgBus = newPicoActivityTestTurn(t, "pico", false)
	al.publishPicoToolResult(context.Background(), ts, "call_1", "exec", ToolResultStatusOK, "x", 0, false)
	expectNoOutbound(t, msgBus)
}

func TestSpawnSubTurn_PublishesPicoSubTurnLifecycleWithLabel(t *testing.T) {
	al, parentTS, msgBus := newPicoActivityTestTurn(t, "pico", true)

	ctx := tools.WithToolCallID(context.Background(), "call_spawn_1")
	_, err := spawnSubTurn(ctx, al, parentTS, SubTurnConfig{
		Model:        "test-model",
		SystemPrompt: "do the thing",
		Label:        "research docs",
	})
	if err != nil {
		t.Fatalf("spawnSubTurn() error = %v", err)
	}

	var phases []PicoSubTurn
	for len(phases) < 2 {
		msg := readOutbound(t, msgBus)
		if msg.Context.Raw[metadataKeyMessageKind] != messageKindSubTurn {
			continue
		}
		var st PicoSubTurn
		if err := json.Unmarshal([]byte(msg.Context.Raw[metadataKeySubTurn]), &st); err != nil {
			t.Fatal(err)
		}
		if msg.ChatID != "pico:session-1" {
			t.Fatalf("subturn sent to chat %q", msg.ChatID)
		}
		phases = append(phases, st)
	}

	spawn, end := phases[0], phases[1]
	if spawn.Phase != SubTurnPhaseSpawn || end.Phase != SubTurnPhaseEnd {
		t.Fatalf("phases = %q, %q", spawn.Phase, end.Phase)
	}
	for _, st := range phases {
		if st.Label != "research docs" {
			t.Fatalf("label = %q, want human label", st.Label)
		}
		if !strings.HasPrefix(st.ChildTurnID, "subturn-") || st.ParentTurnID != "turn-1" {
			t.Fatalf("ids: child=%q parent=%q", st.ChildTurnID, st.ParentTurnID)
		}
		if st.ToolCallID != "call_spawn_1" || st.Depth != 1 || st.TimestampMilli == 0 {
			t.Fatalf("tool_call_id=%q depth=%d ts=%d", st.ToolCallID, st.Depth, st.TimestampMilli)
		}
	}
	if end.Status != "completed" {
		t.Fatalf("end status = %q", end.Status)
	}
}

func TestSpawnSubTurn_LabelFallsBackToChildID(t *testing.T) {
	al, parentTS, msgBus := newPicoActivityTestTurn(t, "pico", true)
	if _, err := spawnSubTurn(context.Background(), al, parentTS, SubTurnConfig{Model: "test-model"}); err != nil {
		t.Fatal(err)
	}
	for {
		msg := readOutbound(t, msgBus)
		if msg.Context.Raw[metadataKeyMessageKind] != messageKindSubTurn {
			continue
		}
		var st PicoSubTurn
		_ = json.Unmarshal([]byte(msg.Context.Raw[metadataKeySubTurn]), &st)
		if st.Label != st.ChildTurnID || st.ToolCallID != "" {
			t.Fatalf("label = %q child = %q tool_call_id = %q", st.Label, st.ChildTurnID, st.ToolCallID)
		}
		return
	}
}

func TestEventMeta_CarriesParentTurnID(t *testing.T) {
	_, ts, _ := newPicoActivityTestTurn(t, "pico", true)
	ts.parentTurnID = "turn-0"
	meta := ts.eventMeta("test", "test")
	if meta.TurnID != "turn-1" || meta.ParentTurnID != "turn-0" {
		t.Fatalf("meta turn=%q parent=%q", meta.TurnID, meta.ParentTurnID)
	}
}

func TestPicoTurnRaw_CarriesTurnIDAndTokenTotals(t *testing.T) {
	ts := &turnState{turnID: "main-turn-9"}
	raw := picoTurnRaw(ts, map[string]string{"x": "1"})
	if raw[metadataKeyTurnID] != "main-turn-9" || raw["x"] != "1" {
		t.Fatalf("raw = %#v", raw)
	}
	if _, ok := raw[metadataKeyTurnInputTokens]; ok {
		t.Fatalf("token keys should be absent before any usage: %#v", raw)
	}
	ts.SetLastUsage(&providers.UsageInfo{PromptTokens: 500, CompletionTokens: 20})
	ts.SetLastUsage(&providers.UsageInfo{PromptTokens: 700, CompletionTokens: 30})
	raw = picoTurnRaw(ts, nil)
	if raw[metadataKeyTurnInputTokens] != "1200" || raw[metadataKeyTurnOutputTokens] != "50" {
		t.Fatalf("raw = %#v", raw)
	}
}

func TestSubTurnEndReason(t *testing.T) {
	cases := []struct {
		name   string
		err    error
		ctxErr error
		limit  string
		want   string
	}{
		{"finished", nil, nil, "", ""},
		{"iteration limit", nil, nil, SubTurnReasonMaxToolIterations, SubTurnReasonMaxToolIterations},
		{"timeout error", fmt.Errorf("llm: %w", context.DeadlineExceeded), nil, "", SubTurnReasonTimeout},
		{"timeout without error", nil, context.DeadlineExceeded, "", SubTurnReasonTimeout},
		{"canceled is not a timeout", nil, context.Canceled, "", ""},
		{"other error keeps its text", errors.New("boom"), nil, "", "boom"},
	}
	for _, tc := range cases {
		if got := subTurnEndReason(tc.err, tc.ctxErr, tc.limit); got != tc.want {
			t.Errorf("%s: reason = %q, want %q", tc.name, got, tc.want)
		}
	}
}

// readSubTurnEnd returns the first subturn end frame published on msgBus.
func readSubTurnEnd(t *testing.T, msgBus *bus.MessageBus) PicoSubTurn {
	t.Helper()
	for {
		msg := readOutbound(t, msgBus)
		if msg.Context.Raw[metadataKeyMessageKind] != messageKindSubTurn {
			continue
		}
		var st PicoSubTurn
		if err := json.Unmarshal([]byte(msg.Context.Raw[metadataKeySubTurn]), &st); err != nil {
			t.Fatal(err)
		}
		if st.Phase == SubTurnPhaseEnd {
			return st
		}
	}
}

func TestSpawnSubTurn_EndSaysWhenTheChildHitMaxToolIterations(t *testing.T) {
	cfg := &config.Config{Agents: config.AgentsConfig{Defaults: config.AgentDefaults{
		Workspace: t.TempDir(), ModelName: "test-model", MaxTokens: 4096, MaxToolIterations: 2,
	}}}
	msgBus := bus.NewMessageBus()
	al := NewAgentLoop(cfg, msgBus, &toolLimitOnlyProvider{})
	al.RegisterTool(&toolLimitTestTool{})
	parentTS := &turnState{
		ctx: context.Background(), turnID: "turn-1", agentID: "main", channel: "pico",
		chatID: "pico:session-1", sessionKey: "agent:main:pico:session-1",
		opts:           processOptions{AllowInterimPicoPublish: true},
		pendingResults: make(chan *tools.ToolResult, 4),
		concurrencySem: make(chan struct{}, testMaxConcurrentSubTurns),
		session:        &ephemeralSessionStore{}, agent: al.registry.GetDefaultAgent(),
	}

	if _, err := spawnSubTurn(context.Background(), al, parentTS, SubTurnConfig{Model: "test-model"}); err != nil {
		t.Fatalf("spawnSubTurn() error = %v", err)
	}
	end := readSubTurnEnd(t, msgBus)
	if end.Status != "completed" || end.Reason != SubTurnReasonMaxToolIterations {
		t.Fatalf(
			"end status = %q reason = %q, want completed / %s",
			end.Status,
			end.Reason,
			SubTurnReasonMaxToolIterations,
		)
	}
}

type blockingUntilDoneProvider struct{}

func (m *blockingUntilDoneProvider) Chat(
	ctx context.Context,
	_ []providers.Message,
	_ []providers.ToolDefinition,
	_ string,
	_ map[string]any,
) (*providers.LLMResponse, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

func (m *blockingUntilDoneProvider) GetDefaultModel() string { return "blocking-model" }

func TestSpawnSubTurn_EndSaysWhenTheChildTimedOut(t *testing.T) {
	cfg := &config.Config{Agents: config.AgentsConfig{Defaults: config.AgentDefaults{
		Workspace: t.TempDir(), ModelName: "test-model", MaxTokens: 4096, MaxToolIterations: 2,
	}}}
	msgBus := bus.NewMessageBus()
	al := NewAgentLoop(cfg, msgBus, &blockingUntilDoneProvider{})
	parentTS := &turnState{
		ctx: context.Background(), turnID: "turn-1", agentID: "main", channel: "pico",
		chatID: "pico:session-1", sessionKey: "agent:main:pico:session-1",
		opts:           processOptions{AllowInterimPicoPublish: true},
		pendingResults: make(chan *tools.ToolResult, 4),
		concurrencySem: make(chan struct{}, testMaxConcurrentSubTurns),
		session:        &ephemeralSessionStore{}, agent: al.registry.GetDefaultAgent(),
	}

	_, _ = spawnSubTurn(
		context.Background(),
		al,
		parentTS,
		SubTurnConfig{Model: "test-model", Timeout: 50 * time.Millisecond},
	)
	end := readSubTurnEnd(t, msgBus)
	if end.Reason != SubTurnReasonTimeout {
		t.Fatalf("end status = %q reason = %q, want reason %s", end.Status, end.Reason, SubTurnReasonTimeout)
	}
}

// childScriptProvider makes a child call one tool, then answer.
type childScriptProvider struct{ calls int }

func (p *childScriptProvider) Chat(
	_ context.Context,
	_ []providers.Message,
	_ []providers.ToolDefinition,
	_ string,
	_ map[string]any,
) (*providers.LLMResponse, error) {
	p.calls++
	if p.calls == 1 {
		return &providers.LLMResponse{
			ReasoningContent: "I should read it first.",
			ToolCalls: []providers.ToolCall{{
				ID: "call_child_1", Type: "function", Name: "tool_limit_test_tool",
				Arguments: map[string]any{"value": "x"},
			}},
			Usage: &providers.UsageInfo{PromptTokens: 100, CompletionTokens: 10},
		}, nil
	}
	return &providers.LLMResponse{
		Content: "child done",
		Usage:   &providers.UsageInfo{PromptTokens: 150, CompletionTokens: 5},
	}, nil
}

func (p *childScriptProvider) GetDefaultModel() string { return "child-script-model" }

func TestSpawnSubTurn_PublishesChildTranscript(t *testing.T) {
	cfg := &config.Config{Agents: config.AgentsConfig{Defaults: config.AgentDefaults{
		Workspace: t.TempDir(), ModelName: "test-model", MaxTokens: 4096, MaxToolIterations: 5,
	}}}
	msgBus := bus.NewMessageBus()
	al := NewAgentLoop(cfg, msgBus, &childScriptProvider{})
	al.RegisterTool(&toolLimitTestTool{})
	parentTS := &turnState{
		ctx: context.Background(), turnID: "turn-1", agentID: "main", channel: "pico",
		chatID: "pico:session-1", sessionKey: "agent:main:pico:session-1",
		opts:           processOptions{AllowInterimPicoPublish: true},
		pendingResults: make(chan *tools.ToolResult, 4),
		concurrencySem: make(chan struct{}, testMaxConcurrentSubTurns),
		session:        &ephemeralSessionStore{}, agent: al.registry.GetDefaultAgent(),
	}
	parentTS.opts.Dispatch.InboundContext = &bus.InboundContext{Channel: "pico", ChatID: "pico:session-1"}

	if _, err := spawnSubTurn(context.Background(), al, parentTS, SubTurnConfig{Model: "test-model"}); err != nil {
		t.Fatalf("spawnSubTurn() error = %v", err)
	}

	var got []PicoSubTurnMessage
	for {
		msg := readOutbound(t, msgBus)
		if msg.Context.Raw[metadataKeyMessageKind] == messageKindSubTurnMessage {
			var m PicoSubTurnMessage
			if err := json.Unmarshal([]byte(msg.Context.Raw[metadataKeySubTurnMsg]), &m); err != nil {
				t.Fatal(err)
			}
			if msg.ChatID != "pico:session-1" {
				t.Fatalf("subturn message sent to %q", msg.ChatID)
			}
			got = append(got, m)
			continue
		}
		if msg.Context.Raw[metadataKeyMessageKind] == messageKindSubTurn &&
			strings.Contains(msg.Context.Raw[metadataKeySubTurn], `"phase":"end"`) {
			break
		}
	}
	kinds := make([]string, 0, len(got))
	for i, m := range got {
		kinds = append(kinds, m.Kind)
		if m.Seq != i+1 || !strings.HasPrefix(m.ChildTurnID, "subturn-") || m.ParentTurnID != "turn-1" || m.Depth != 1 {
			t.Fatalf("frame %d ids/seq: %+v", i, m)
		}
	}
	want := []string{SubTurnMessageThought, SubTurnMessageToolCalls, SubTurnMessageToolResult, SubTurnMessageText}
	if strings.Join(kinds, ",") != strings.Join(want, ",") {
		t.Fatalf("kinds = %v, want %v", kinds, want)
	}
	if got[1].ToolCalls[0].ID != "call_child_1" || got[2].ToolCallID != "call_child_1" ||
		got[2].Status != ToolResultStatusOK {
		t.Fatalf("call/result pairing: %+v / %+v", got[1], got[2])
	}
	if got[3].Content != "child done" || got[3].TurnUsage == nil || got[3].TurnUsage.InputTokens != 250 {
		t.Fatalf("final text frame = %+v", got[3])
	}
}

func TestPublishPicoSubTurnMessage_CapsFramesPerChild(t *testing.T) {
	al, msgBus := newSubTurnMessageTestLoop(t)
	child := &turnState{
		turnID: "subturn-9", parentTurnID: "turn-1", depth: 1, agentID: "main",
		channel: "pico", chatID: "pico:session-1", agent: al.registry.GetDefaultAgent(),
	}
	go func() {
		for i := 0; i < picoSubTurnMessageCap+5; i++ {
			al.publishPicoSubTurnMessage(child, PicoSubTurnMessage{Kind: SubTurnMessageText, Content: "x"})
		}
	}()
	var last PicoSubTurnMessage
	for n := 0; n < picoSubTurnMessageCap+1; n++ {
		msg := readOutbound(t, msgBus)
		_ = json.Unmarshal([]byte(msg.Context.Raw[metadataKeySubTurnMsg]), &last)
	}
	if last.Kind != SubTurnMessageOmitted || last.Seq != picoSubTurnMessageCap+1 {
		t.Fatalf("frame after the cap = %+v, want one omitted notice", last)
	}
	expectNoOutbound(t, msgBus)
}

func newSubTurnMessageTestLoop(t *testing.T) (*AgentLoop, *bus.MessageBus) {
	t.Helper()
	al, _, msgBus, _, cleanup := newTestAgentLoop(t)
	t.Cleanup(cleanup)
	return al, msgBus
}

// budgetBurnProvider asks for a tool every call and reports 600 tokens per call.
type budgetBurnProvider struct{ calls int }

func (p *budgetBurnProvider) Chat(
	_ context.Context,
	_ []providers.Message,
	tools []providers.ToolDefinition,
	_ string,
	_ map[string]any,
) (*providers.LLMResponse, error) {
	p.calls++
	usage := &providers.UsageInfo{PromptTokens: 500, CompletionTokens: 100}
	if len(tools) == 0 {
		return &providers.LLMResponse{Content: "partial answer", Usage: usage}, nil
	}
	return &providers.LLMResponse{
		ToolCalls: []providers.ToolCall{{
			ID: fmt.Sprintf("call_%d", p.calls), Type: "function", Name: "tool_limit_test_tool",
			Arguments: map[string]any{"value": "x"},
		}},
		Usage: usage,
	}, nil
}

func (p *budgetBurnProvider) GetDefaultModel() string { return "budget-burn-model" }

func TestSpawnSubTurn_EndSaysWhenTheChildUsedUpItsTokenBudget(t *testing.T) {
	cfg := &config.Config{Agents: config.AgentsConfig{Defaults: config.AgentDefaults{
		Workspace: t.TempDir(), ModelName: "test-model", MaxTokens: 4096, MaxToolIterations: 10,
		SubTurn: config.SubTurnConfig{DefaultTokenBudget: 1000},
	}}}
	msgBus := bus.NewMessageBus()
	provider := &budgetBurnProvider{}
	al := NewAgentLoop(cfg, msgBus, provider)
	al.RegisterTool(&toolLimitTestTool{})
	parentTS := &turnState{
		ctx: context.Background(), turnID: "turn-1", agentID: "main", channel: "pico",
		chatID: "pico:session-1", sessionKey: "agent:main:pico:session-1",
		opts:           processOptions{AllowInterimPicoPublish: true},
		pendingResults: make(chan *tools.ToolResult, 4),
		concurrencySem: make(chan struct{}, testMaxConcurrentSubTurns),
		session:        &ephemeralSessionStore{}, agent: al.registry.GetDefaultAgent(),
	}

	if _, err := spawnSubTurn(context.Background(), al, parentTS, SubTurnConfig{Model: "test-model"}); err != nil {
		t.Fatalf("spawnSubTurn() error = %v", err)
	}
	end := readSubTurnEnd(t, msgBus)
	if end.Reason != SubTurnReasonTokenBudget || end.Status != "completed" {
		t.Fatalf("end status = %q reason = %q, want completed / %s", end.Status, end.Reason, SubTurnReasonTokenBudget)
	}
	// 600 tokens per call against 1000: two calls with tools, then one last call without them.
	if provider.calls != 3 {
		t.Fatalf("model calls = %d, want 3", provider.calls)
	}
}
