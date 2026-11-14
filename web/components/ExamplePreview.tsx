export function ExamplePreview({ scanId }: { scanId: string }) {
  return (
    // Static renders of the same GLB parts and palette used by the editor.
    // eslint-disable-next-line @next/next/no-img-element
    <img src={`/examples/${encodeURIComponent(scanId)}.webp`} alt="" width={1600} height={960} loading="lazy" className="example-model-preview" />
  );
}
