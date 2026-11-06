package agent

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/sipeed/picoclaw/pkg/bus"
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
