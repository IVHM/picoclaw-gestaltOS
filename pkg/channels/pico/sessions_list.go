package pico

import (
	"context"
	"time"

	"github.com/sipeed/picoclaw/pkg/channels"
)

const (
	sessionsListDefaultLimit = 50
	sessionsListMaxLimit     = 200
	sessionsListTimeout      = 5 * time.Second
)

// SetSessionLister lets the channel answer sessions.list from its clients.
func (c *PicoChannel) SetSessionLister(l channels.SessionLister) {
	c.connsMu.Lock()
	defer c.connsMu.Unlock()
	c.sessionLister = l
}

// handleSessionsList answers a sessions.list request on the asking connection
// only. Read only; the socket's token already authorized the client.
func (c *PicoChannel) handleSessionsList(pc *picoConn, msg PicoMessage) {
	c.connsMu.RLock()
	lister := c.sessionLister
	c.connsMu.RUnlock()
	if lister == nil {
		reply := newError("not_supported", "session listing is not available")
		reply.ID = msg.ID
		pc.writeJSON(reply)
		return
	}

	limit := sessionsListDefaultLimit
	if v, ok := msg.Payload["limit"].(float64); ok && v > 0 {
		limit = min(int(v), sessionsListMaxLimit)
	}
	ctx, cancel := context.WithTimeout(c.ctx, sessionsListTimeout)
	defer cancel()
	list, err := lister(ctx, c.Name())
	if err != nil {
		reply := newError("sessions_list_failed", err.Error())
		reply.ID = msg.ID
		pc.writeJSON(reply)
		return
	}

	total := len(list)
	if len(list) > limit {
		list = list[:limit]
	}
	sessions := make([]map[string]any, 0, len(list))
	for _, s := range list {
		item := map[string]any{
			"id":            s.ID,
			"message_count": s.MessageCount,
			"busy":          s.Busy,
		}
		if s.Title != "" {
			item["title"] = s.Title
		}
		if !s.LastActive.IsZero() {
			item["last_active"] = s.LastActive.UnixMilli()
		}
		sessions = append(sessions, item)
	}
	reply := newMessage(TypeSessionsListResult, map[string]any{"sessions": sessions, "total": total})
	reply.ID = msg.ID
	pc.writeJSON(reply)
}
