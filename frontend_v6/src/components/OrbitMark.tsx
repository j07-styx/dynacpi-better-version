export function OrbitMark({ active }: { active?: boolean }) {
  return (
    <svg width="72" height="72" viewBox="0 0 72 72" aria-hidden="true">
      <g className={active ? "anim-orbit" : undefined} style={{ transformOrigin: "36px 36px" }}>
        <ellipse cx="36" cy="36" rx="30" ry="11" fill="none" stroke="var(--orbit)" />
        <circle cx="66" cy="36" r="2.5" fill="var(--primary)" />
      </g>
      <ellipse cx="36" cy="36" rx="30" ry="11" fill="none" stroke="var(--orbit)" transform="rotate(60 36 36)" />
      <ellipse cx="36" cy="36" rx="30" ry="11" fill="none" stroke="var(--orbit)" transform="rotate(-60 36 36)" />
      <circle cx="36" cy="36" r="4" fill="var(--primary)" />
    </svg>
  );
}