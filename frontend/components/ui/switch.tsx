"use client";
import * as SwitchPrimitive from "@radix-ui/react-switch";
import * as React from "react";
import { cn } from "@/lib/utils";

export function Switch({ className, ...p }: React.ComponentProps<typeof SwitchPrimitive.Root>) {
  return (
    <SwitchPrimitive.Root
      className={cn(
        "relative inline-flex h-4 w-7 shrink-0 cursor-pointer items-center rounded-full border border-border bg-panel-2 transition-colors data-[state=checked]:bg-accent disabled:opacity-50",
        className,
      )}
      {...p}
    >
      <SwitchPrimitive.Thumb className="block size-3 translate-x-0.5 rounded-full bg-fg transition-transform data-[state=checked]:translate-x-3.5 data-[state=checked]:bg-black" />
    </SwitchPrimitive.Root>
  );
}
