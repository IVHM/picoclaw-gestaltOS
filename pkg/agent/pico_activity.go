// PicoClaw - Ultra-lightweight personal AI agent

package agent

import (
	"context"
	"encoding/json"
	"errors"
	"strconv"
	"strings"
	"time"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/logger"
	"github.com/sipeed/picoclaw/pkg/utils"
)

// Pico activity messages are structured side-channel events for live clients
// (tool results and sub-turn lifecycle). They ride the outbound bus like
// tool_calls messages, carry their data as JSON in Context.Raw, and are only
// published for turns that belong to the pico channel.
const (
	messageKindToolResult = "tool_result"
	messageKindSubTurn    = "subturn"
	metadataKeyToolResult = "tool_result"
	metadataKeySubTurn    = "subturn"
	metadataKeyTurnID     = "turn_id"
	// Prompt and completion tokens summed over the turn so far (all LLM calls,
	// tool loop included). Sent on tool_calls and on the final answer.
	metadataKeyTurnInputTokens  = "turn_input_tokens"
	metadataKeyTurnOutputTokens = "turn_output_tokens"

	// picoToolResultMaxContentRunes caps the result text sent to the socket.
	// The full result stays in the session store; content_len reports its size.
	picoToolResultMaxContentRunes = 4000

	ToolResultStatusOK      = "ok"
	ToolResultStatusError   = "error"
	ToolResultStatusDenied  = "denied"
	ToolResultStatusSkipped = "skipped"

	SubTurnPhaseSpawn           = "spawn"
	SubTurnPhaseEnd             = "end"
	SubTurnPhaseResultDelivered = "result_delivered"
	SubTurnPhaseOrphan          = "orphan"
)

// PicoToolResult is the tool_result payload sent on the pico socket. It pairs
// with an earlier tool_calls message through ToolCallID.
type PicoToolResult struct {
	ToolCallID   string `json:"tool_call_id"`
	Tool         string `json:"tool,omitempty"`
	Status       string `json:"status"`
	IsError      bool   `json:"is_error"`
	Content      string `json:"content"`
	ContentLen   int    `json:"content_len"`
	Truncated    bool   `json:"truncated,omitempty"`
	Async        bool   `json:"async,omitempty"`
	DurationMs   int64  `json:"duration_ms"`
	TurnID       string `json:"turn_id,omitempty"`
	ParentTurnID string `json:"parent_turn_id,omitempty"`
}

// PicoSubTurn is the subturn payload sent on the pico socket to the parent
// turn's chat.
type PicoSubTurn struct {
	Phase          string `json:"phase"`
	ChildTurnID    string `json:"child_turn_id"`
	ParentTurnID   string `json:"parent_turn_id,omitempty"`
	AgentID        string `json:"agent_id,omitempty"`
	Label          string `json:"label,omitempty"`
	ToolCallID     string `json:"tool_call_id,omitempty"`
	Async          bool   `json:"async,omitempty"`
	Depth          int    `json:"depth,omitempty"`
	Status         string `json:"status,omitempty"`
	Reason         string `json:"reason,omitempty"`
	DurationMs     int64  `json:"duration_ms,omitempty"`
	InputTokens    int    `json:"input_tokens,omitempty"`
	OutputTokens   int    `json:"output_tokens,omitempty"`
	ResultLen      int    `json:"result_len,omitempty"`
	TimestampMilli int64  `json:"ts"`
}

func picoToolResultStatus(isError bool) string {
	if isError {
		return ToolResultStatusError
	}
	return ToolResultStatusOK
}

// shouldPublishPicoToolResult mirrors the gate on tool_calls messages, so a
// client never receives a result for a call it was not shown. Sub-turns have
// AllowInterimPicoPublish=false and are reported through subturn messages.
func shouldPublishPicoToolResult(ts *turnState) bool {
	return ts != nil && ts.channel == "pico" && ts.chatID != "" && ts.opts.AllowInterimPicoPublish
}

func (al *AgentLoop) publishPicoToolResult(
	ctx context.Context,
	ts *turnState,
	toolCallID string,
	toolName string,
	status string,
	content string,
	duration time.Duration,
	async bool,
) {
	if al == nil || al.bus == nil || !shouldPublishPicoToolResult(ts) || strings.TrimSpace(toolCallID) == "" {
		return
	}

	clipped := utils.Truncate(content, picoToolResultMaxContentRunes)
	snap := ts.snapshot()
	payload := PicoToolResult{
		ToolCallID:   toolCallID,
		Tool:         toolName,
		Status:       status,
		IsError:      status != ToolResultStatusOK,
		Content:      clipped,
		ContentLen:   len([]rune(content)),
		Truncated:    clipped != content,
		Async:        async,
		DurationMs:   duration.Milliseconds(),
		TurnID:       snap.TurnID,
		ParentTurnID: snap.ParentTurnID,
	}
	al.publishPicoActivity(ctx, ts, messageKindToolResult, metadataKeyToolResult, payload)
}

// publishPicoSubTurn sends a sub-turn lifecycle message to the parent turn's
// pico chat. The child shares the parent's chat, so this is where its row and
// the running count come from.
func (al *AgentLoop) publishPicoSubTurn(parentTS *turnState, payload PicoSubTurn) {
	if al == nil || al.bus == nil || parentTS == nil || parentTS.channel != "pico" || parentTS.chatID == "" {
		return
	}
	payload.TimestampMilli = time.Now().UnixMilli()
	// Sub-turn events can fire after the parent context is done (orphans,
	// critical children), so they publish on a fresh context.
	al.publishPicoActivity(context.Background(), parentTS, messageKindSubTurn, metadataKeySubTurn, payload)
}

func (al *AgentLoop) publishPicoActivity(
	ctx context.Context,
	ts *turnState,
	kind string,
	rawKey string,
	payload any,
) {
	if ts == nil || ts.agent == nil {
		return
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		logger.WarnCF("agent", "Failed to serialize pico activity", map[string]any{
			"kind":  kind,
			"error": err.Error(),
		})
		return
	}
	msg := outboundMessageForTurnWithOptions(ts, "", outboundTurnMessageOptions{
		kind: kind,
		raw:  map[string]string{rawKey: string(encoded)},
	})

	pubCtx, pubCancel := context.WithTimeout(ctx, 3*time.Second)
	err = al.bus.PublishOutbound(pubCtx, msg)
	pubCancel()
	if err != nil && !errors.Is(err, context.DeadlineExceeded) &&
		!errors.Is(err, context.Canceled) &&
		!errors.Is(err, bus.ErrBusClosed) {
		logger.WarnCF("agent", "Failed to publish pico activity", map[string]any{
			"kind":    kind,
			"channel": ts.channel,
			"chat_id": ts.chatID,
			"error":   err.Error(),
		})
	}
}

// picoTurnRaw is the outbound metadata that ties a pico message to its turn:
// the turn id and the turn's token totals so far. Token keys are left out
// while both counts are zero.
func picoTurnRaw(ts *turnState, extra map[string]string) map[string]string {
	raw := make(map[string]string, len(extra)+3)
	for k, v := range extra {
		raw[k] = v
	}
	if ts == nil {
		return raw
	}
	raw[metadataKeyTurnID] = ts.snapshot().TurnID
	if in, out := ts.TokenTotals(); in > 0 || out > 0 {
		raw[metadataKeyTurnInputTokens] = strconv.Itoa(in)
		raw[metadataKeyTurnOutputTokens] = strconv.Itoa(out)
	}
	return raw
}
