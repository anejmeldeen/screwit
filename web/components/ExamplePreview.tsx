export function ExamplePreview({ scanId }: { scanId: string }) {
  const trap = scanId === "scn_trap";
  return <svg viewBox="0 0 500 260" aria-hidden="true" className="example-drawing">
    <defs><pattern id={`grid-${scanId}`} width="25" height="25" patternUnits="userSpaceOnUse"><path d="M25 0H0V25" fill="none" stroke="currentColor" opacity=".08" /></pattern></defs>
    <rect width="500" height="260" fill={`url(#grid-${scanId})`} />
    <g fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round">
      {trap ? <>
        <path d="M205 45v113c0 67 103 67 103 0v-34h40" strokeWidth="30" stroke="#d5c899" />
        <path d="M190 45v113c0 88 133 88 133 0v-19h25M220 45v113c0 47 73 47 73 0v-49h55" />
        <ellipse cx="205" cy="45" rx="15" ry="5" fill="#efe4bd" />
        <path d="M184 98h42v19h-42zM287 139h42v19h-42z" fill="#e8d16f" />
        <path d="M194 102v11m10-11v11m10-11v11m83 30v11m10-11v11m10-11v11" opacity=".6" />
        <path d="M160 45v165m-5-165h10m-10 165h10M183 232h146m-146-5v10m146-10v10" opacity=".3" strokeDasharray="4 4" />
      </> : <>
        <path d="M173 116v48c0 45 154 45 154 0v-48" fill="#d5c899" />
        <ellipse cx="250" cy="116" rx="77" ry="42" fill="#e8d16f" />
        <ellipse cx="250" cy="116" rx="54" ry="29" />
        <path d="M231 105v26h38v-26z" fill="#343123" />
        <path d="M185 139v31m19-21v32m22-25v32m24-30v32m24-32v30m22-36v31m19-41v30" opacity=".45" />
        <path d="M157 217h186m-186-5v10m186-10v10M353 77v112m-5-112h10m-10 112h10" opacity=".3" strokeDasharray="4 4" />
      </>}
    </g>
    <text x="24" y="29" fontSize="10" letterSpacing="2" fill="currentColor" opacity=".55">{trap ? "01 / PIPE ASSEMBLY" : "02 / CONTROL KNOB"}</text>
    <text x="24" y="239" fontSize="11" fill="currentColor" opacity=".55">Illustration · open to explore the scan</text>
  </svg>;
}
