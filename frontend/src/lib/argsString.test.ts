import { describe, expect, it } from "vitest"

import { formatArgsString, isValidArgsString, parseArgsString } from "./argsString"

describe("parseArgsString", () => {
  it("parses k=v pairs", () => {
    expect(parseArgsString("a=1,b=two")).toEqual([
      { key: "a", value: "1" },
      { key: "b", value: "two" },
    ])
  })

  it("returns [] for empty input", () => {
    expect(parseArgsString("")).toEqual([])
    expect(parseArgsString("   ")).toEqual([])
  })

  it("keeps values with slashes and dots", () => {
    expect(
      parseArgsString("pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=auto")
    ).toEqual([
      { key: "pretrained", value: "meta-llama/Llama-3.1-8B-Instruct" },
      { key: "dtype", value: "auto" },
    ])
  })

  it("rejects malformed input", () => {
    expect(parseArgsString("novalue")).toBeNull()
    expect(parseArgsString("=x")).toBeNull()
    expect(parseArgsString("a=1,b")).toBeNull()
    expect(parseArgsString("a=b=c")).toBeNull()
  })
})

describe("formatArgsString", () => {
  it("round-trips", () => {
    const s = "retriever_type=bm25,bm25_index_path=msmarco-v1-passage"
    expect(formatArgsString(parseArgsString(s)!)).toBe(s)
  })
})

describe("isValidArgsString", () => {
  it("accepts valid and empty, rejects malformed", () => {
    expect(isValidArgsString("model=gpt-4o")).toBe(true)
    expect(isValidArgsString("")).toBe(true)
    expect(isValidArgsString("oops")).toBe(false)
  })
})
