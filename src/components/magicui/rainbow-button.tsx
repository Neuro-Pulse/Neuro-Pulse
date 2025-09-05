"use client";

import React, { useState } from "react";
import { cn } from "@/lib/utils";

interface RainbowButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  showCopyIcon?: boolean;
  copyText?: string;
  variant?: "default" | "outline";
  size?: "default" | "sm" | "lg";
  children: React.ReactNode;
}

export function RainbowButton({
  showCopyIcon = false,
  copyText = "",
  variant = "default",
  size = "default",
  className,
  children,
  onClick,
  title,
  ...props
}: RainbowButtonProps) {
  const [copied, setCopied] = useState(false);

  const handleClick = async (e: React.MouseEvent<HTMLButtonElement>) => {
    // Si hay copyText y showCopyIcon, copiar al clipboard
    if (showCopyIcon && copyText) {
      try {
        await navigator.clipboard.writeText(copyText);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } catch (err) {
        console.error('Failed to copy text: ', err);
      }
    }
    
    // Ejecutar el onClick original si existe
    if (onClick) {
      onClick(e);
    }
  };

  const sizeClasses = {
    default: "px-6 py-3 text-base",
    sm: "px-4 py-2 text-sm",
    lg: "px-8 py-4 text-lg"
  };

  return (
    <button
      className={cn(
        // Base styles
        "relative inline-flex items-center justify-center rounded-lg font-medium transition-all duration-300 ease-out",
        "bg-gradient-to-r from-purple-500 via-violet-500 to-pink-500",
        "hover:from-purple-600 hover:via-violet-600 hover:to-pink-600",
        "text-white shadow-lg hover:shadow-xl",
        "focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2",
        "active:scale-95",
        // Size classes
        sizeClasses[size],
        // Custom className
        className
      )}
      onClick={handleClick}
      title={title}
      {...props}
    >
      <span className="relative z-10 flex items-center gap-2">
        {children}
        {showCopyIcon && (
          <span className="ml-2">
            {copied ? (
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
            ) : (
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path d="M8 3a1 1 0 011-1h2a1 1 0 110 2H9a1 1 0 01-1-1z" />
                <path d="M6 3a2 2 0 00-2 2v11a2 2 0 002 2h8a2 2 0 002-2V5a2 2 0 00-2-2 3 3 0 01-3 3H9a3 3 0 01-3-3z" />
              </svg>
            )}
          </span>
        )}
      </span>
      
      {/* Rainbow animation overlay */}
      <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-purple-400 via-violet-400 to-pink-400 opacity-0 hover:opacity-20 transition-opacity duration-300" />
    </button>
  );
}
