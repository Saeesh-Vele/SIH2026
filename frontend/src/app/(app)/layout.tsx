import { AuthProvider } from "@/components/auth/auth-provider";
import { TooltipProvider } from "@/components/ui/tooltip";

/**
 * Everything behind sign-in, plus the sign-in page itself. Firebase and the
 * tooltip provider load here rather than in the root layout, so the landing
 * page ships neither.
 */
export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <AuthProvider>
      <TooltipProvider delayDuration={200}>{children}</TooltipProvider>
    </AuthProvider>
  );
}
