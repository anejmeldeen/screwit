type NamedBuild = { scan_id: string; title: string };

// Reserve "~" for disambiguation; titles can never generate this separator.
export function workspaceSlugs(builds: NamedBuild[]) {
  const bases = builds.map(build => build.title.normalize("NFKC").toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, "-").replace(/^-|-$/g, "") || "workspace");
  const counts = new Map<string, number>();
  for (const base of bases) counts.set(base, (counts.get(base) ?? 0) + 1);
  return new Map(builds.map((build, index) => {
    const base = bases[index];
    const slug = base === "new" || counts.get(base)! > 1
      ? base + "~" + build.scan_id : base;
    return [build.scan_id, slug];
  }));
}
