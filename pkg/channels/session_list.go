package channels

import (
	"context"
	"time"
)

// SessionSummary is one chat session as a channel lists it to its clients.
type SessionSummary struct {
	// ID is the channel's own session id: for pico, the session_id a client
	// connects with.
	ID string `json:"id"`
	// Title is the session's first user message, clipped.
	Title string `json:"title,omitempty"`
	// MessageCount is the number of stored messages.
	MessageCount int `json:"message_count"`
	// LastActive is when the session was last written.
	LastActive time.Time `json:"-"`
	// Busy is true while a top-level turn of the session is running.
	Busy bool `json:"busy"`
}

// SessionLister lists the sessions of one channel, most recently active first.
type SessionLister func(ctx context.Context, channel string) ([]SessionSummary, error)

type sessionListerSetter interface {
	SetSessionLister(l SessionLister)
}

// SetSessionLister hands the lister to every channel that can answer session
// list requests from its clients.
func (m *Manager) SetSessionLister(l SessionLister) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.sessionLister = l
	for _, ch := range m.channels {
		if setter, ok := ch.(sessionListerSetter); ok {
			setter.SetSessionLister(l)
		}
	}
}
