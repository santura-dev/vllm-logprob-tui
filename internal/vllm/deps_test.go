package vllm

import (
	"os"
	"regexp"
	"testing"
)

func TestCharmStackMajorVersions(t *testing.T) {
	gomod, err := os.ReadFile("../../go.mod")
	if err != nil {
		t.Fatal(err)
	}
	for _, dep := range []string{"bubbles", "bubbletea", "lipgloss"} {
		re := regexp.MustCompile(`github\.com/charmbracelet/` + dep + ` v(\d+)`)
		m := re.FindSubmatch(gomod)
		if m == nil {
			t.Errorf("%s missing from go.mod", dep)
			continue
		}
		if string(m[1]) != "1" {
			t.Errorf("charmbracelet/%s major version = %s, pinned to v1", dep, m[1])
		}
	}
}
