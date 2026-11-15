"use client";

import { createContext } from "react";
import type { Part } from "@/lib/contracts.gen";

export const ManualContext = createContext<{ name: string; parts: Part[] }>({
  name: "Workspace", parts: [],
});
