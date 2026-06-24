import React, { useMemo } from 'react';

import { SceneComponentProps, sceneGraph, SceneObjectBase } from '@grafana/scenes';
import { shareDashboardType } from 'app/features/dashboard/components/ShareModal/utils';
import { SceneShareTabState } from 'app/features/dashboard-scene/sharing/types';

import { SharePDFBase } from '../SharePDF';
import { toReportSceneVariables } from '../utils/variables';

export class SharePDFTab extends SceneObjectBase<SceneShareTabState> {
  public tabId = shareDashboardType.pdf;
  static Component = SharePDFTabRenderer;

  public getTabLabel() {
    return 'PDF';
  }
}

function SharePDFTabRenderer({ model }: SceneComponentProps<SharePDFTab>) {
  const { modalRef, dashboardRef } = model.useState();
  const dashboard = dashboardRef.resolve();

  const variables = useMemo(() => {
    const sceneGraphVars = sceneGraph.getVariables(dashboard).state.variables;
    return toReportSceneVariables(sceneGraphVars);
  }, [dashboard]);

  return (
    <SharePDFBase
      onDismiss={modalRef?.resolve().onDismiss}
      displayQueryVariablesAlert={false}
      isSnapshot={Boolean(dashboard.state.meta.isSnapshot)}
      variables={variables}
      dashboardUid={dashboard.state.uid!}
    />
  );
}
