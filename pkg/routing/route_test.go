package routing

import (
	"testing"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/config"
)

func testConfig(agents []config.AgentConfig) *config.Config {
	return &config.Config{
		Agents: config.AgentsConfig{
			Defaults: config.AgentDefaults{
				Workspace: "/tmp/picoclaw-test",
				ModelName: "gpt-4",
			},
			List: agents,
		},
		Session: config.SessionConfig{
			Dimensions: []string{"sender"},
		},
	}
}

func TestResolveRoute_DefaultAgent_NoBindings(t *testing.T) {
	cfg := testConfig(nil)
	r := NewRouteResolver(cfg)

	route := r.ResolveRoute(bus.InboundContext{
		Channel:  "telegram",
		ChatType: "direct",
		SenderID: "user1",
	})

	if route.AgentID != DefaultAgentID {
		t.Errorf("AgentID = %q, want %q", route.AgentID, DefaultAgentID)
	}
	if route.MatchedBy != "default" {
		t.Errorf("MatchedBy = %q, want 'default'", route.MatchedBy)
	}
	if len(route.SessionPolicy.Dimensions) != 1 || route.SessionPolicy.Dimensions[0] != "sender" {
		t.Errorf("SessionPolicy.Dimensions = %v, want [sender]", route.SessionPolicy.Dimensions)
	}
	if route.SessionPolicy.IdentityLinks != nil {
		t.Errorf("SessionPolicy.IdentityLinks = %v, want nil", route.SessionPolicy.IdentityLinks)
	}
}

func TestResolveRoute_UsesNormalizedInboundContextFields(t *testing.T) {
	cfg := testConfig([]config.AgentConfig{{ID: "sales", Default: true}})
	r := NewRouteResolver(cfg)

	route := r.ResolveRoute(bus.InboundContext{
		Channel:  "Telegram",
		Account:  "Bot2",
		ChatType: "direct",
		SenderID: "user123",
	})

	if route.AgentID != "sales" {
		t.Errorf("AgentID = %q, want 'sales'", route.AgentID)
	}
	if route.Channel != "telegram" {
		t.Errorf("Channel = %q, want 'telegram'", route.Channel)
	}
	if route.AccountID != "bot2" {
		t.Errorf("AccountID = %q, want 'bot2'", route.AccountID)
	}
	if route.MatchedBy != "default" {
		t.Errorf("MatchedBy = %q, want 'default'", route.MatchedBy)
	}
}

func TestResolveRoute_DispatchFirstMatchWins(t *testing.T) {
	cfg := testConfig([]config.AgentConfig{
		{ID: "main", Default: true},
		{ID: "support"},
		{ID: "sales"},
	})
	cfg.Agents.Dispatch = &config.DispatchConfig{
		Rules: []config.DispatchRule{
			{
				Name:  "support-group",
				Agent: "support",
				When: config.DispatchSelector{
					Channel: "telegram",
					Chat:    "group:-100123",
				},
			},
			{
				Name:  "vip-in-group",
				Agent: "sales",
				When: config.DispatchSelector{
					Channel: "telegram",
					Chat:    "group:-100123",
					Sender:  "12345",
				},
			},
		},
	}
	r := NewRouteResolver(cfg)

	route := r.ResolveRoute(bus.InboundContext{
		Channel:  "telegram",
		ChatID:   "-100123",
		ChatType: "group",
		SenderID: "12345",
	})

	if route.AgentID != "support" {
		t.Fatalf("AgentID = %q, want support", route.AgentID)
	}
	if route.MatchedBy != "dispatch.rule:support-group" {
		t.Fatalf("MatchedBy = %q, want dispatch.rule:support-group", route.MatchedBy)
	}
}

func TestResolveRoute_DispatchOverridesSessionDimensions(t *testing.T) {
	cfg := testConfig([]config.AgentConfig{
		{ID: "main", Default: true},
		{ID: "support"},
	})
	cfg.Session.Dimensions = []string{"chat"}
	cfg.Agents.Dispatch = &config.DispatchConfig{
		Rules: []config.DispatchRule{
			{
				Name:  "support-dm",
				Agent: "support",
				When: config.DispatchSelector{
					Channel: "telegram",
					Chat:    "direct:user-1",
				},
				SessionDimensions: []string{"chat", "sender"},
			},
		},
	}
	r := NewRouteResolver(cfg)

	route := r.ResolveRoute(bus.InboundContext{
		Channel:  "telegram",
		ChatID:   "user-1",
		ChatType: "direct",
		SenderID: "user-1",
	})

	if route.AgentID != "support" {
		t.Fatalf("AgentID = %q, want support", route.AgentID)
	}
	if got := route.SessionPolicy.Dimensions; len(got) != 2 || got[0] != "chat" || got[1] != "sender" {
		t.Fatalf("SessionPolicy.Dimensions = %v, want [chat sender]", got)
	}
}

func TestResolveRoute_DispatchMentionedRule(t *testing.T) {
	cfg := testConfig([]config.AgentConfig{
		{ID: "main", Default: true},
		{ID: "support"},
	})
	mentioned := true
	cfg.Agents.Dispatch = &config.DispatchConfig{
		Rules: []config.DispatchRule{
			{
				Name:  "slack-mentions",
				Agent: "support",
				When: config.DispatchSelector{
					Channel:   "slack",
					Space:     "workspace:t001",
					Mentioned: &mentioned,
				},
			},
		},
	}
	r := NewRouteResolver(cfg)

	route := r.ResolveRoute(bus.InboundContext{
		Channel:   "slack",
		ChatID:    "C123",
		ChatType:  "channel",
		SpaceID:   "T001",
		SpaceType: "workspace",
		SenderID:  "U123",
		Mentioned: true,
	})

	if route.AgentID != "support" {
		t.Fatalf("AgentID = %q, want support", route.AgentID)
	}
}

func TestResolveRoute_InvalidAgentFallsToDefault(t *testing.T) {
	agents := []config.AgentConfig{
		{ID: "main", Default: true},
	}
	cfg := testConfig(agents)
	r := NewRouteResolver(cfg)

	route := r.ResolveRoute(bus.InboundContext{Channel: "telegram"})

	if route.AgentID != "main" {
		t.Errorf("AgentID = %q, want 'main' (invalid agent should fall to default)", route.AgentID)
	}
}

func TestResolveRoute_DefaultAgentSelection(t *testing.T) {
	agents := []config.AgentConfig{
		{ID: "alpha"},
		{ID: "beta", Default: true},
		{ID: "gamma"},
	}
	cfg := testConfig(agents)
	r := NewRouteResolver(cfg)

	route := r.ResolveRoute(bus.InboundContext{Channel: "cli"})

	if route.AgentID != "beta" {
		t.Errorf("AgentID = %q, want 'beta' (marked as default)", route.AgentID)
	}
}

func TestResolveRoute_NoDefaultUsesFirst(t *testing.T) {
	agents := []config.AgentConfig{
		{ID: "alpha"},
		{ID: "beta"},
	}
	cfg := testConfig(agents)
	r := NewRouteResolver(cfg)

	route := r.ResolveRoute(bus.InboundContext{Channel: "cli"})

	if route.AgentID != "alpha" {
		t.Errorf("AgentID = %q, want 'alpha' (first in list)", route.AgentID)
	}
}

func TestResolveRoute_DispatchChatPrefix(t *testing.T) {
	cfg := testConfig([]config.AgentConfig{
		{ID: "main", Default: true},
		{ID: "assistant"},
	})
	cfg.Agents.Dispatch = &config.DispatchConfig{
		Rules: []config.DispatchRule{{
			Name:  "project-assistants",
			Agent: "assistant",
			When:  config.DispatchSelector{Channel: "pico", Chat: "direct:pico:assist-*"},
		}},
	}
	r := NewRouteResolver(cfg)

	for chat, want := range map[string]string{
		"pico:assist-web-alpha": "assistant",
		"pico:Assist-X":         "assistant",
		"pico:proj-web-alpha":   "main",
		"pico:assist":           "main",
	} {
		route := r.ResolveRoute(bus.InboundContext{Channel: "pico", ChatID: chat, ChatType: "direct", SenderID: "pico-user"})
		if route.AgentID != want {
			t.Errorf("chat %q: AgentID = %q, want %q", chat, route.AgentID, want)
		}
	}
}

func TestSelectorValueMatches(t *testing.T) {
	cases := []struct {
		want, got string
		ok        bool
	}{
		{"direct:pico:a", "direct:pico:a", true},
		{"direct:pico:a", "direct:pico:ab", false},
		{"direct:pico:a*", "direct:pico:ab", true},
		{"direct:pico:a*", "direct:pico:a", true},
		{"*", "anything", true},
		{"direct:pico:a*", "group:pico:ab", false},
	}
	for _, c := range cases {
		if got := selectorValueMatches(c.want, c.got); got != c.ok {
			t.Errorf("selectorValueMatches(%q, %q) = %v, want %v", c.want, c.got, got, c.ok)
		}
	}
}
