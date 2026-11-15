import type { ReactNode } from "react";

export type DashboardIconName = "dashboard" | "box" | "settings" | "plus" | "search" | "arrow" | "logout" | "menu" | "camera";

const shapes: Record<DashboardIconName, ReactNode> = {
  dashboard: <><rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="7" rx="1.5" /><rect x="3" y="14" width="7" height="7" rx="1.5" /><rect x="14" y="14" width="7" height="7" rx="1.5" /></>,
  box: <><path d="m12 3 9 5v8l-9 5-9-5V8l9-5Z" /><path d="m3 8 9 5 9-5M12 13v8M7.5 5.5l9 5" /></>,
  settings: <><path d="m9 3-.7 2.4-2.4 1L3.7 6l-2 3.4 1.7 1.8v2l-1.7 1.7 2 3.5 2.3-.5 2.3 1L9 21h4l.7-2.1 2.4-1 2.2.5 2-3.5-1.7-1.7v-2l1.7-1.8-2-3.4-2.2.4-2.4-1L13 3H9Z" transform="translate(1 0)" /><circle cx="12" cy="12" r="3" /></>,
  plus: <path d="M12 5v14M5 12h14" />,
  search: <><circle cx="10.5" cy="10.5" r="6.5" /><path d="m16 16 5 5" /></>,
  arrow: <path d="M5 12h14m-6-6 6 6-6 6" />,
  logout: <path d="M9 4H4v16h5m6-13 5 5-5 5m-6-5h11" />,
  menu: <path d="M4 6h16M4 12h16M4 18h16" />,
  camera: <><rect x="3" y="5" width="13" height="14" rx="2" /><path d="m16 10 5-3v10l-5-3" /></>,
};

export function DashboardIcon({ name, className = "" }: { name: DashboardIconName; className?: string }) {
  return <svg className={className} width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">{shapes[name]}</svg>;
}
