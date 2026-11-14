import { Workspace } from "@/components/Workspace";
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
      <Workspace scanId={id} initialPartId={partId} />
    </>
  );
}
