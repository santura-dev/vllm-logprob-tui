package system

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"sync"

	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/mem"
)

// Stats is a snapshot of host resources.
type Stats struct {
	GPU        string
	CPUPercent float64
	RAMUsedMB  uint64
	RAMTotalMB uint64
	RAMPercent float64
}

var (
	gpuMu        sync.Mutex
	gpuAvailable = true // checked once; most hosts never have nvidia-smi
)

// Collect gathers GPU (via nvidia-smi), CPU, and RAM stats.
// Missing nvidia-smi is checked once and cached; GPU reports "not available".
func Collect() Stats {
	s := Stats{GPU: "not available"}

	gpuMu.Lock()
	hasGPU := gpuAvailable
	gpuMu.Unlock()
	if hasGPU {
		s.GPU = gpuStats()
	}

	if pct, err := cpu.Percent(0, false); err == nil && len(pct) > 0 {
		s.CPUPercent = pct[0]
	}
	if info, err := mem.VirtualMemory(); err == nil {
		s.RAMUsedMB = info.Used / 1024 / 1024
		s.RAMTotalMB = info.Total / 1024 / 1024
		s.RAMPercent = info.UsedPercent
	}
	return s
}

func gpuStats() string {
	out, err := exec.Command("nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits").Output()
	if err != nil {
		gpuMu.Lock()
		gpuAvailable = false
		gpuMu.Unlock()
		return "not available"
	}
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	if len(lines) == 0 {
		return "not available"
	}
	parts := strings.Split(lines[0], ", ")
	if len(parts) < 5 {
		return "not available"
	}
	used, _ := strconv.Atoi(strings.TrimSpace(parts[0]))
	total, _ := strconv.Atoi(strings.TrimSpace(parts[1]))
	util, _ := strconv.Atoi(strings.TrimSpace(parts[2]))
	temp, _ := strconv.Atoi(strings.TrimSpace(parts[3]))
	power, _ := strconv.ParseFloat(strings.TrimSpace(parts[4]), 64)
	return fmt.Sprintf("%d/%d MB, %d%% util, %d°C, %.1f W", used, total, util, temp, power)
}
