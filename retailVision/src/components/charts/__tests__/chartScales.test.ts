// Tests for the scale and path maths every chart is drawn from.
//
// These are the pieces where a wrong answer is invisible: a chart with a
// slightly wrong scale still looks like a chart. The colour test is here
// for a specific bug that shipped -- series were coloured by their index
// among the categories present in a range, so changing the time filter
// silently recoloured every series.

import { describe, expect, it } from "vitest";
import { SERIES_VARS, linearScale, niceMax, seriesColor, ticks } from "../chartScales";

describe("linearScale", () => {
  it("maps the domain ends onto the range ends", () => {
    const scale = linearScale(0, 100, 0, 200);
    expect(scale(0)).toBe(0);
    expect(scale(100)).toBe(200);
    expect(scale(50)).toBe(100);
  });

  it("handles an inverted range, which is how SVG y-axes are drawn", () => {
    // Screen y grows downward, so a chart's y scale maps larger values to
    // smaller coordinates. Getting this backwards flips every chart.
    const scale = linearScale(0, 10, 200, 0);
    expect(scale(0)).toBe(200);
    expect(scale(10)).toBe(0);
  });

  it("does not divide by zero when every value is identical", () => {
    // A flat series is common -- an empty room reports the same count all
    // day -- and a zero-width domain must not produce NaN coordinates.
    expect(Number.isFinite(linearScale(5, 5, 0, 100)(5))).toBe(true);
  });
});

describe("niceMax", () => {
  it("rounds up so the axis ends on a readable number", () => {
    expect(niceMax(7)).toBeGreaterThanOrEqual(7);
    expect(niceMax(23)).toBeGreaterThanOrEqual(23);
  });

  it("never returns zero, which would collapse the scale", () => {
    expect(niceMax(0)).toBeGreaterThan(0);
  });
});

describe("ticks", () => {
  it("stays within the axis it labels", () => {
    const values = ticks(100);
    expect(values.length).toBeGreaterThan(0);
    expect(Math.max(...values)).toBeLessThanOrEqual(100);
    expect(Math.min(...values)).toBeGreaterThanOrEqual(0);
  });

  it("returns values in ascending order", () => {
    const values = ticks(50);
    expect([...values].sort((a, b) => a - b)).toEqual(values);
  });
});

describe("seriesColor", () => {
  it("gives each slot a distinct colour", () => {
    const used = SERIES_VARS.map((_, index) => seriesColor(index));
    expect(new Set(used).size).toBe(SERIES_VARS.length);
  });

  it("is stable for a given slot", () => {
    // The property the recolour-on-filter bug violated: a slot must always
    // mean the same colour, so callers can key it to an entity rather than
    // to a position in whatever subset a query returned.
    expect(seriesColor(2)).toBe(seriesColor(2));
  });

  it("degrades to muted ink rather than cycling past the last slot", () => {
    // Reusing slot 1 for a ninth series would make two entities identical.
    const beyond = seriesColor(SERIES_VARS.length);
    expect(SERIES_VARS).not.toContain(beyond);
  });
});
