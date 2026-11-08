package pico

import (
	"context"
	"testing"
	"time"

	"github.com/sipeed/picoclaw/pkg/channels"
)

func TestHandleSessionsList(t *testing.T) {
	ch := newTestPicoChannel(t)
	if err := ch.Start(context.Background()); err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	defer ch.Stop(context.Background())
	clientConn, received, cleanup := newTestPicoWebSocket(t)
	defer cleanup()
	pc := &picoConn{id: "conn-1", conn: clientConn, sessionID: "sess-1"}
	ch.addConnForTest(pc)

	read := func() PicoMessage {
		t.Helper()
		select {
		case msg := <-received:
			return msg
		case <-time.After(time.Second):
			t.Fatal("no reply")
		}
		return PicoMessage{}
	}

	ch.handleMessage(pc, PicoMessage{Type: TypeSessionsList, ID: "r0"})
	if msg := read(); msg.Type != TypeError || msg.ID != "r0" || msg.Payload["code"] != "not_supported" {
		t.Fatalf("without a lister: %+v", msg)
	}

	var asked string
	ch.SetSessionLister(func(_ context.Context, channel string) ([]channels.SessionSummary, error) {
		asked = channel
		return []channels.SessionSummary{
			{
				ID:           "demo",
				Title:        "audit my workspace",
				MessageCount: 12,
				LastActive:   time.UnixMilli(1794118000000),
				Busy:         true,
			},
			{ID: "old", MessageCount: 3},
		}, nil
	})
	ch.handleMessage(pc, PicoMessage{Type: TypeSessionsList, ID: "r1", Payload: map[string]any{"limit": float64(1)}})
	msg := read()
	if msg.Type != TypeSessionsListResult || msg.ID != "r1" || asked != "pico" {
		t.Fatalf("reply = %+v (asked for %q)", msg, asked)
	}
	list, _ := msg.Payload["sessions"].([]any)
	if len(list) != 1 || msg.Payload["total"] != float64(2) {
		t.Fatalf("limit/total: %+v", msg.Payload)
	}
	first, _ := list[0].(map[string]any)
	if first["id"] != "demo" || first["title"] != "audit my workspace" || first["message_count"] != float64(12) ||
		first["busy"] != true || first["last_active"] != float64(1794118000000) {
		t.Fatalf("item = %#v", first)
	}
}
