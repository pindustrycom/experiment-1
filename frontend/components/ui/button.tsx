import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import * as React from "react";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-1.5 whitespace-nowrap rounded-md text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60 disabled:pointer-events-none disabled:opacity-50 [&_svg]:size-3.5",
  {
    variants: {
      variant: {
        default: "bg-accent text-black hover:bg-accent/85",
        secondary: "bg-panel-2 text-fg border border-border hover:bg-border/60",
        outline: "border border-border bg-transparent hover:bg-panel-2",
        ghost: "hover:bg-panel-2 text-muted hover:text-fg",
        destructive: "bg-risk text-white hover:bg-risk/85",
      },
      size: { default: "h-8 px-3", sm: "h-7 px-2", lg: "h-9 px-4 text-sm", icon: "h-8 w-8" },
    },
    defaultVariants: { variant: "default", size: "default" },
  },
);

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(({ className, variant, size, asChild = false, ...props }, ref) => {
  const Comp = asChild ? Slot : "button";
  return <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />;
});
Button.displayName = "Button";
