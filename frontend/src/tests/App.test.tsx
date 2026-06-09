import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import App from "../App";

describe("App", () => {
  it("renders both independent workflows", () => {
    render(<App />);

    expect(screen.getByRole("heading", {name: "圖片生成 Prompt"})).toBeInTheDocument();
    expect(screen.getByRole("heading", {name: "Prompt 生成圖片"})).toBeInTheDocument();
    expect(screen.getByRole("heading", {name: "Prompt Refinement Loop"})).toBeInTheDocument();
  });
});
