import * as React from "react";
import { cn } from "@/lib/utils";

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(({ className, ...p }, ref) => (
  <input
    ref={ref}
    className={cn(
      "h-8 w-full rounded-md border border-border bg-panel-2 px-2.5 text-xs text-fg placeholder:text-muted focus:outline-none focus:ring-2 focus:ring-accent/50",
      className,
    )}
    {...p}
  />
));
Input.displayName = "Input";

export const Select = React.forwardRef<HTMLSelectElement, React.SelectHTMLAttributes<HTMLSelectElement>>(({ className, ...p }, ref) => (
  <select
    ref={ref}
    className={cn("h-8 rounded-md border border-border bg-panel-2 px-2 text-xs text-fg focus:outline-none focus:ring-2 focus:ring-accent/50", className)}
    {...p}
  />
));
Select.displayName = "Select";

export function Label({ className, ...p }: React.LabelHTMLAttributes<HTMLLabelElement>) {
  return <label className={cn("mb-1 block text-[11px] font-medium text-muted", className)} {...p} />;
}

export const Textarea = React.forwardRef<HTMLTextAreaElement, React.TextareaHTMLAttributes<HTMLTextAreaElement>>(({ className, ...p }, ref) => (
  <textarea
    ref={ref}
    className={cn("w-full rounded-md border border-border bg-panel-2 p-2 text-xs text-fg focus:outline-none focus:ring-2 focus:ring-accent/50", className)}
    {...p}
  />
));
Textarea.displayName = "Textarea";
