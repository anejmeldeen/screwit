import { Workspace } from "@/components/Workspace";
import { ViewerPreloadLinks } from "@/components/ViewerPreloadLinks";

export default async function ScanPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return (
    <>
      <ViewerPreloadLinks />
      <Workspace scanId={id} />
    </>
  );
}
