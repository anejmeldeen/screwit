import { NamedWorkspace } from "@/components/NamedWorkspace";
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
      <NamedWorkspace key={id} segment={id} />
    </>
  );
}
