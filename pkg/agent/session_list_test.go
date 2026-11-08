package agent

import (
	"context"
	"testing"

	"github.com/sipeed/picoclaw/pkg/providers"
)

func TestListChannelSessions_ListsPicoSessionsWithTitles(t *testing.T) {
	al, cfg, _, _, cleanup := newTestAgentLoop(t)
	defer cleanup()
	// The default config keeps one session per chat; the minimal test config does not.
	cfg.Session.Dimensions = []string{"chat"}
	ctx := context.Background()
	for _, chat := range []string{"pico:demo", "pico:other"} {
		if _, err := al.ProcessDirectWithChannel(ctx, "audit   my\nworkspace "+chat, "", "pico", chat); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := al.ProcessDirectWithChannel(ctx, "not pico", "", "telegram", "123"); err != nil {
		t.Fatal(err)
	}

	list, err := al.ListChannelSessions(ctx, "pico")
	if err != nil {
		t.Fatal(err)
	}
	if len(list) != 2 {
		t.Fatalf("sessions = %+v, want the 2 pico ones", list)
	}
	ids := map[string]string{}
	for _, s := range list {
		ids[s.ID] = s.Title
		if s.MessageCount == 0 || s.LastActive.IsZero() || s.Busy {
			t.Fatalf("summary = %+v", s)
		}
	}
	if ids["demo"] == "" || ids["other"] == "" {
		t.Fatalf("ids/titles = %v", ids)
	}
	if list[0].LastActive.Before(list[1].LastActive) {
		t.Fatal("not sorted most recent first")
	}
}

func TestSessionTitleAndID(t *testing.T) {
	if got := channelSessionID("pico", "direct:pico:demo"); got != "demo" {
		t.Fatalf("channelSessionID = %q", got)
	}
	title := sessionTitle(
		[]providers.Message{{Role: "system", Content: "x"}, {Role: "user", Content: " hello \n  there "}},
	)
	if title != "hello there" {
		t.Fatalf("title = %q", title)
	}
}
