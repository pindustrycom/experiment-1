import { cva, type VariantProps } from "class-variance-authority";
import * as React from "react";
import { cn } from "@/lib/utils";

const badgeVariants = cva("inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide border", {
  variants: {
    variant: {
      default: "border-border bg-panel-2 text-muted",
      accent: "border-accent/40 bg-accent/10 text-accent",
      blue: "border-accent-2/40 bg-accent-2/10 text-accent-2",
      risk: "border-risk/40 bg-risk/10 text-risk",
      warn: "border-warn/40 bg-warn/10 text-warn",
      ok: "border-ok/40 bg-ok/10 text-ok",
      sim: "border-sim/50 bg-sim/10 text-sim",
    },
  },
  defaultVariants: { variant: "default" },
});

export function Badge({ className, variant, ...p }: React.HTMLAttributes<HTMLSpanElement> & VariantProps<typeof badgeVariants>) {
  return <span className={cn(badgeVariants({ variant }), className)} {...p} />;
}
