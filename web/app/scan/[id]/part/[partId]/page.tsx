import { NamedWorkspace } from "@/components/NamedWorkspace";
import { ViewerPreloadLinks } from "@/components/ViewerPreloadLinks";

// Deep link that focuses the viewer on one part (SPEC 2.1).
export default async function ScanPartPage({
  params,
}: {
  params: Promise<{ id: string; partId: string }>;
}) {
  const { id, partId } = await params;
  return (
    <>
      <ViewerPreloadLinks />
      <NamedWorkspace key={JSON.stringify([id, partId])} segment={id} partId={partId} />
    </>
  );
}
