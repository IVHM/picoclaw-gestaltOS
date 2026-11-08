// PicoClaw - Ultra-lightweight personal AI agent

package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/logger"
	"github.com/sipeed/picoclaw/pkg/providers"
	"github.com/sipeed/picoclaw/pkg/utils"
)

// Pico activity messages are structured side-channel events for live clients
// (tool results and sub-turn lifecycle). They ride the outbound bus like
// tool_calls messages, carry their data as JSON in Context.Raw, and are only
// published for turns that belong to the pico channel.
const (
	messageKindToolResult = "tool_result"
	messageKindSubTurn    = "subturn"
	// messageKindSubTurnMessage carries one step of a child's own transcript.
	messageKindSubTurnMessage = "subturn_message"
	metadataKeyToolResult     = "tool_result"
	metadataKeySubTurn        = "subturn"
	metadataKeySubTurnMsg     = "subturn_message"
	metadataKeyTurnID         = "turn_id"
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

	// SubTurnMessageThought and the kinds below are the values of a
	// subturn.message frame's kind.
	SubTurnMessageThought    = "thought"
	SubTurnMessageText       = "text"
	SubTurnMessageToolCalls  = "tool_calls"
	SubTurnMessageToolResult = "tool_result"
	SubTurnMessageOmitted    = "omitted"

	// picoSubTurnMessageCap is how many transcript frames one child may send.
	// After it, one "omitted" frame is sent and the rest are dropped.
	picoSubTurnMessageCap = 200

	// SubTurnReasonMaxToolIterations is the end frame's reason when the child
	// ran out of tool iterations instead of finishing.
	SubTurnReasonMaxToolIterations = "max_tool_iterations"
	// SubTurnReasonTimeout is the end frame's reason when the child's timeout
	// expired. Other errors keep their error text.
	SubTurnReasonTimeout = "timeout"
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

// PicoTurnUsage is a running token total, as in turn_usage.
type PicoTurnUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// PicoSubTurnMessage is one step of a child's own transcript, sent to the
// parent's pico chat as it happens (subturn.message).
type PicoSubTurnMessage struct {
	ChildTurnID    string                  `json:"child_turn_id"`
	ParentTurnID   string                  `json:"parent_turn_id,omitempty"`
	Depth          int                     `json:"depth,omitempty"`
	Seq            int                     `json:"seq"`
	Kind           string                  `json:"kind"`
	Content        string                  `json:"content,omitempty"`
	ContentLen     int                     `json:"content_len,omitempty"`
	Truncated      bool                    `json:"truncated,omitempty"`
	ToolCalls      []utils.VisibleToolCall `json:"tool_calls,omitempty"`
	ToolCallID     string                  `json:"tool_call_id,omitempty"`
	Tool           string                  `json:"tool,omitempty"`
	Status         string                  `json:"status,omitempty"`
	IsError        bool                    `json:"is_error,omitempty"`
	DurationMs     int64                   `json:"duration_ms,omitempty"`
	ModelName      string                  `json:"model_name,omitempty"`
	TurnUsage      *PicoTurnUsage          `json:"turn_usage,omitempty"`
	TimestampMilli int64                   `json:"ts"`
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
	if al == nil || al.bus == nil || strings.TrimSpace(toolCallID) == "" {
		return
	}
	clipped := utils.Truncate(content, picoToolResultMaxContentRunes)
	if isPicoSubTurn(ts) {
		al.publishPicoSubTurnMessage(ts, PicoSubTurnMessage{
			Kind:       SubTurnMessageToolResult,
			Content:    clipped,
			ContentLen: len([]rune(content)),
			Truncated:  clipped != content,
			ToolCallID: toolCallID,
			Tool:       toolName,
			Status:     status,
			IsError:    status != ToolResultStatusOK,
			DurationMs: duration.Milliseconds(),
		})
		return
	}
	if !shouldPublishPicoToolResult(ts) {
		return
	}

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

// subTurnEndReason explains why a child stopped: a limit it hit, a timeout,
// the error text, or "" when it finished on its own.
func subTurnEndReason(err, ctxErr error, limit string) string {
	switch {
	case errors.Is(err, context.DeadlineExceeded) || (err == nil && errors.Is(ctxErr, context.DeadlineExceeded)):
		return SubTurnReasonTimeout
	case err != nil:
		return err.Error()
	default:
		return limit
	}
}

// isPicoSubTurn reports a child turn whose root chat is a pico chat.
func isPicoSubTurn(ts *turnState) bool {
	return ts != nil && ts.parentTurnID != "" && ts.channel == "pico" && ts.chatID != ""
}

// publishPicoSubTurnLLMResponse sends a child's model response as up to three
// subturn.message frames: its thought, its text and its tool calls.
func (al *AgentLoop) publishPicoSubTurnLLMResponse(
	ts *turnState,
	reasoning, content, modelName string,
	toolCalls []providers.ToolCall,
) {
	if !isPicoSubTurn(ts) {
		return
	}
	for _, step := range []struct{ kind, text string }{
		{SubTurnMessageThought, reasoning},
		{SubTurnMessageText, content},
	} {
		if strings.TrimSpace(step.text) == "" {
			continue
		}
		clipped := utils.Truncate(step.text, picoToolResultMaxContentRunes)
		al.publishPicoSubTurnMessage(ts, PicoSubTurnMessage{
			Kind: step.kind, Content: clipped, ContentLen: len([]rune(step.text)),
			Truncated: clipped != step.text, ModelName: modelName,
		})
	}
	if len(toolCalls) > 0 {
		calls := utils.BuildVisibleToolCalls(toolCalls, al.cfg.Agents.Defaults.GetToolFeedbackMaxArgsLength())
		if len(calls) > 0 {
			al.publishPicoSubTurnMessage(ts, PicoSubTurnMessage{
				Kind: SubTurnMessageToolCalls, ToolCalls: calls, ModelName: modelName,
			})
		}
	}
}

// publishPicoSubTurnMessage stamps a child transcript frame (ids, sequence,
// running token totals) and sends it, up to picoSubTurnMessageCap per child.
func (al *AgentLoop) publishPicoSubTurnMessage(ts *turnState, msg PicoSubTurnMessage) {
	if al == nil || al.bus == nil || !isPicoSubTurn(ts) {
		return
	}
	seq := ts.nextSubTurnMessageSeq()
	switch {
	case seq > picoSubTurnMessageCap+1:
		return
	case seq == picoSubTurnMessageCap+1:
		msg = PicoSubTurnMessage{
			Kind:    SubTurnMessageOmitted,
			Content: fmt.Sprintf("further messages of this sub-turn are not sent (cap %d)", picoSubTurnMessageCap),
		}
	}
	snap := ts.snapshot()
	msg.ChildTurnID = ts.subTurnID
	if msg.ChildTurnID == "" {
		msg.ChildTurnID = snap.TurnID
	}
	msg.ParentTurnID = snap.ParentTurnID
	msg.Depth = ts.depth
	msg.Seq = seq
	if in, out := ts.TokenTotals(); in > 0 || out > 0 {
		msg.TurnUsage = &PicoTurnUsage{InputTokens: in, OutputTokens: out, TotalTokens: in + out}
	}
	msg.TimestampMilli = time.Now().UnixMilli()
	// Children can outlive their parent's context; publish on a fresh one.
	al.publishPicoActivity(context.Background(), ts, messageKindSubTurnMessage, metadataKeySubTurnMsg, msg)
}
