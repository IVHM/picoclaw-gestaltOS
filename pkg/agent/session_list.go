package agent

import (
	"context"
	"sort"
	"strings"
	"time"

	"github.com/sipeed/picoclaw/pkg/channels"
	"github.com/sipeed/picoclaw/pkg/providers"
	"github.com/sipeed/picoclaw/pkg/session"
	"github.com/sipeed/picoclaw/pkg/utils"
)

// sessionTitleMaxRunes clips a listed session's title (its first user message).
const sessionTitleMaxRunes = 80

// sessionInfoStore is what listing needs from a session store beyond
// SessionStore: the structured scope and the metadata summary.
type sessionInfoStore interface {
	GetSessionScope(sessionKey string) *session.SessionScope
	GetSessionInfo(sessionKey string) (count int, updatedAt time.Time, ok bool)
}

// ListChannelSessions lists every stored session of a channel across all
// agents, most recently active first. Read only.
func (al *AgentLoop) ListChannelSessions(ctx context.Context, channel string) ([]channels.SessionSummary, error) {
	registry := al.GetRegistry()
	if registry == nil {
		return nil, nil
	}
	busy := al.busySessionKeys()
	var out []channels.SessionSummary
	seen := map[string]bool{}
	for _, agentID := range registry.ListAgentIDs() {
		agent, ok := registry.GetAgent(agentID)
		if !ok || agent == nil || agent.Sessions == nil {
			continue
		}
		store, ok := agent.Sessions.(sessionInfoStore)
		if !ok {
			continue
		}
		for _, key := range agent.Sessions.ListSessions() {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			scope := store.GetSessionScope(key)
			if scope == nil || scope.Channel != channel {
				continue
			}
			id := channelSessionID(channel, scope.Values["chat"])
			if id == "" || seen[id] {
				continue
			}
			count, updated, _ := store.GetSessionInfo(key)
			if count == 0 {
				continue
			}
			seen[id] = true
			out = append(out, channels.SessionSummary{
				ID:           id,
				Title:        sessionTitle(agent.Sessions.GetHistory(key)),
				MessageCount: count,
				LastActive:   updated,
				Busy:         busy[key],
			})
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].LastActive.After(out[j].LastActive) })
	return out, nil
}

// busySessionKeys is the set of session keys with a top-level turn running.
func (al *AgentLoop) busySessionKeys() map[string]bool {
	busy := map[string]bool{}
	al.activeTurnStates.Range(func(_, value any) bool {
		if ts, ok := value.(*turnState); ok && ts.parentTurnID == "" && ts.sessionKey != "" {
			busy[ts.sessionKey] = true
		}
		return true
	})
	return busy
}

// channelSessionID turns a scope's chat value ("direct:pico:demo") into the id
// a client of that channel uses ("demo").
func channelSessionID(channel, chat string) string {
	chat = strings.TrimPrefix(chat, "direct:")
	chat = strings.TrimPrefix(chat, "group:")
	return strings.TrimPrefix(chat, channel+":")
}

// sessionTitle is the first user message of a history, on one line, clipped.
func sessionTitle(history []providers.Message) string {
	for _, msg := range history {
		if msg.Role == "user" && strings.TrimSpace(msg.Content) != "" {
			return utils.Truncate(strings.Join(strings.Fields(msg.Content), " "), sessionTitleMaxRunes)
		}
	}
	return ""
}
