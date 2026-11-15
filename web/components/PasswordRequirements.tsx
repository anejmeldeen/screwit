import { DashboardIcon } from "./DashboardIcon";

export interface PasswordRule { label: string; met: boolean; problem: string }

// What a new password must satisfy, shared by sign-up and Settings. The server accepts 8 to 128
// characters (the inputs stop at 128, so only the minimum is listed). Leaving the username out
// follows NIST SP 800-63B's advice against passwords built from context words such as the username.
// ``current`` and ``confirm`` add the two rules that only apply when changing a password.
export function passwordRules({ password, username, current, confirm }: {
  password: string; username: string; current?: string; confirm?: string;
}): PasswordRule[] {
  const name = username.trim().toLowerCase();
  const lower = password.toLowerCase();
  const usesName = name.length >= 3 ? lower.includes(name) : name.length > 0 && lower === name;
  const rules: PasswordRule[] = [
    { label: "At least 8 characters", met: password.length >= 8, problem: "Use at least 8 characters." },
    { label: "Doesn't include your username", met: password.length > 0 && !usesName,
      problem: "Choose a password that doesn't include your username." },
  ];
  if (current !== undefined) rules.push({
    label: "Different from your current password", met: password.length > 0 && current.length > 0 && password !== current,
    problem: "Choose a password that's different from your current one.",
  });
  if (confirm !== undefined) rules.push({
    label: "New passwords match", met: password.length > 0 && password === confirm, problem: "The new passwords must match.",
  });
  return rules;
}

// The rules as a checklist under the password field: each one ticks as soon as the password meets it.
// Screen readers hear the state through the hidden "Done"/"Not yet" text when the field describes
// itself with this list (aria-describedby), without a live region chattering on every keystroke.
export function PasswordRequirements({ id, rules }: { id: string; rules: PasswordRule[] }) {
  return <ul id={id} className="password-rules" aria-label="Password requirements">
    {rules.map(rule => <li key={rule.label} className={rule.met ? "is-met" : undefined}>
      <span className="password-rule-mark" aria-hidden="true">{rule.met && <DashboardIcon name="check" />}</span>
      <span className="sr-only">{rule.met ? "Done: " : "Not yet: "}</span>{rule.label}
    </li>)}
  </ul>;
}
