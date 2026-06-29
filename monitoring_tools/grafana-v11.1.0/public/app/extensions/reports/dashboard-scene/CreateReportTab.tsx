import React from 'react';

import { SceneComponentProps, SceneObjectBase } from '@grafana/scenes';
import { CreateReportTab as CreateReportTabBase } from 'app/extensions/reports/CreateReportTab';
import { shareDashboardType } from 'app/features/dashboard/components/ShareModal/utils';
import { SceneShareTabState } from 'app/features/dashboard-scene/sharing/types';

export class CreateReportTab extends SceneObjectBase<SceneShareTabState> {
  public tabId = shareDashboardType.report;
  static Component = CreateReportTabRenderer;

  public getTabLabel() {
    return 'Report';
  }
}

function CreateReportTabRenderer({ model }: SceneComponentProps<CreateReportTab>) {
  const { dashboardRef, modalRef } = model.useState();
  const dashboard = dashboardRef.resolve();

  return (
    <CreateReportTabBase
      dashboard={dashboard}
      onDismiss={() => {
        modalRef?.resolve()?.onDismiss();
      }}
    />
  );
}
