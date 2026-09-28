"use client";

import { createContext, useContext, useEffect, useMemo, useState } from "react";
import {
  EmailAuthProvider,
  GoogleAuthProvider,
  createUserWithEmailAndPassword,
  linkWithCredential,
  linkWithPopup,
  onAuthStateChanged,
  sendPasswordResetEmail,
  signInAnonymously,
  signInWithEmailAndPassword,
  signInWithPopup,
  signOut as firebaseSignOut,
  type Auth,
  type User,
} from "firebase/auth";
import { firebaseConfigured, getFirebaseAuth } from "@/lib/firebase";

interface AuthContextValue {
  user: User | null;
  /** True until Firebase has said whether a session was restored. */
  loading: boolean;
  /** False when the NEXT_PUBLIC_FIREBASE_* variables are missing. */
  configured: boolean;
  signInWithGoogle: () => Promise<void>;
  signInWithEmail: (email: string, password: string) => Promise<void>;
  signUpWithEmail: (email: string, password: string) => Promise<void>;
  sendPasswordReset: (email: string) => Promise<void>;
  continueAsGuest: () => Promise<void>;
  /**
   * Turn the current guest into a full account. Linking keeps the same uid,
   * so every query already run as a guest stays in the history.
   */
  upgradeWithGoogle: () => Promise<void>;
  upgradeWithEmail: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

function requireAuth(): Auth {
  const auth = getFirebaseAuth();
  if (!auth) throw Object.assign(new Error("Firebase is not configured"), { code: "app/no-config" });
  return auth;
}

function requireUser(): User {
  const user = requireAuth().currentUser;
  if (!user) throw Object.assign(new Error("Not signed in"), { code: "app/no-user" });
  return user;
}

const google = () => new GoogleAuthProvider();

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(firebaseConfigured);

  useEffect(() => {
    const auth = getFirebaseAuth();
    if (!auth) return;
    return onAuthStateChanged(auth, (next) => {
      setUser(next);
      setLoading(false);
    });
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      loading,
      configured: firebaseConfigured,
      signInWithGoogle: async () => {
        await signInWithPopup(requireAuth(), google());
      },
      signInWithEmail: async (email, password) => {
        await signInWithEmailAndPassword(requireAuth(), email, password);
      },
      signUpWithEmail: async (email, password) => {
        await createUserWithEmailAndPassword(requireAuth(), email, password);
      },
      sendPasswordReset: async (email) => {
        await sendPasswordResetEmail(requireAuth(), email);
      },
      continueAsGuest: async () => {
        await signInAnonymously(requireAuth());
      },
      upgradeWithGoogle: async () => {
        const { user: linked } = await linkWithPopup(requireUser(), google());
        // onAuthStateChanged does not fire on a link; refresh by hand so the
        // menu stops saying "Guest".
        setUser(linked);
      },
      upgradeWithEmail: async (email, password) => {
        const credential = EmailAuthProvider.credential(email, password);
        const { user: linked } = await linkWithCredential(requireUser(), credential);
        setUser(linked);
      },
      signOut: async () => {
        await firebaseSignOut(requireAuth());
      },
    }),
    [user, loading],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used inside <AuthProvider>");
  return context;
}
