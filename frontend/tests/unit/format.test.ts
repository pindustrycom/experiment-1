import { describe, expect, it } from "vitest";
import { ago, directionLabel, fmtInZone, fmtNum, fmtPct, riskLevel } from "@/lib/format";

describe("format helpers", () => {
  it("never renders a missing value as zero", () => {
    expect(fmtNum(null)).toBe("—");
    expect(fmtNum(undefined)).toBe("—");
    expect(fmtNum(Number.NaN)).toBe("—");
    expect(fmtPct(null)).toBe("—");
    expect(fmtNum(0)).toBe("0.0");
  });

  it("classifies risk relative to the pair threshold", () => {
    expect(riskLevel(null)).toBe("none");
    expect(riskLevel(72, 70)).toBe("high");
    expect(riskLevel(60, 70)).toBe("elevated");
    expect(riskLevel(40, 70)).toBe("normal");
    expect(riskLevel(60, 55)).toBe("high");
  });

  it("labels relative direction with tickers", () => {
    expect(directionLabel("a_advantaged", "KO", "PEP")).toBe("KO advantaged");
    expect(directionLabel("b_advantaged", "KO", "PEP")).toBe("PEP advantaged");
    expect(directionLabel("uncertain", "KO", "PEP")).toBe("Uncertain");
  });

  it("formats relative ages", () => {
    const now = new Date("2026-10-15T12:00:00Z");
    expect(ago(null, now)).toBe("never");
    expect(ago("2026-10-15T11:59:30Z", now)).toBe("just now");
    expect(ago("2026-10-15T11:30:00Z", now)).toBe("30m ago");
    expect(ago("2026-10-15T09:00:00Z", now)).toBe("3h ago");
    expect(ago("2026-10-12T12:00:00Z", now)).toBe("3d ago");
  });

  it("shows scheduler times in New York time across DST", () => {
    expect(fmtInZone("2026-10-15T21:30:00Z", "America/New_York")).toBe("2026-10-15 17:30 EDT");
    expect(fmtInZone("2026-11-16T22:30:00Z", "America/New_York")).toBe("2026-11-16 17:30 EST");
  });
});
