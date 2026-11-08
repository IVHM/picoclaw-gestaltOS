package tools

import "testing"

func TestIsBuiltTaskPrompt(t *testing.T) {
	built := "Task TI-7f3a: audit.\nCriteria: ...\nbundle_id CB-0123456789 (pass this prompt unchanged)\n"
	if !IsBuiltTaskPrompt(built) {
		t.Fatal("a context-builder prompt should be recognized")
	}
	for _, task := range []string{"Inspect these paths.", "bundle_id CB-0123456789\nthen do more", "bundle_id CB-xyz"} {
		if IsBuiltTaskPrompt(task) {
			t.Fatalf("%q is not a built prompt", task)
		}
	}
}
