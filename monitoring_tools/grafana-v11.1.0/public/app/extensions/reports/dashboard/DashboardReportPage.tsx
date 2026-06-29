// Libraries
import { css } from '@emotion/css';
import React, { useEffect, useState } from 'react';

import { GrafanaTheme2 } from '@grafana/data';
import { SceneGridLayout } from '@grafana/scenes';
import { useStyles2 } from '@grafana/ui';
import PageLoader from 'app/core/components/PageLoader/PageLoader';
import { GrafanaRouteComponentProps } from 'app/core/navigation/types';
import { DashboardPageRouteParams } from 'app/features/dashboard/containers/types';
import { getDashboardScenePageStateManager } from 'app/features/dashboard-scene/pages/DashboardScenePageStateManager';
import { DashboardScene } from 'app/features/dashboard-scene/scene/DashboardScene';
import { DashboardRoutes } from 'app/types';

import { ReportGridRenderer } from './ReportGridRenderer';

export interface Props extends GrafanaRouteComponentProps<DashboardPageRouteParams> {}

export function DashboardReportPage({ match, route }: Props) {
  const stateManager = getDashboardScenePageStateManager();
  const { dashboard, isLoading, loadError } = stateManager.useState();

  useEffect(() => {
    stateManager.loadDashboard({ uid: match.params.uid!, route: DashboardRoutes.Report });

    return () => {
      stateManager.clearState();
    };
  }, [stateManager, match.params.uid, route.routeName]);

  if (!dashboard) {
    return (
      <div>
        {isLoading && <PageLoader />}
        {loadError && <h2>{loadError}</h2>}
      </div>
    );
  }

  return <DashbordReportRenderer model={dashboard} />;
}

interface RendererProps {
  model: DashboardScene;
}

function DashbordReportRenderer({ model }: RendererProps) {
  const [isActive, setIsActive] = useState(false);
  const dashboard = model.useState();
  const timeRange = dashboard.$timeRange!.useState();

  const styles = useStyles2(getStyles);
  useEffect(() => {
    setIsActive(true);
    return model.activate();
  }, [model]);

  if (!isActive) {
    return null;
  }

  return (
    <div className={styles.canvas}>
      <div className={styles.body}>
        <ReportGridRenderer grid={dashboard.body as SceneGridLayout} title={dashboard.title} timeRange={timeRange} />
      </div>
    </div>
  );
}

function getStyles(theme: GrafanaTheme2) {
  return {
    canvas: css({
      label: 'canvas',
      display: 'flex',
      flexDirection: 'column',
      flexBasis: '100%',
      flexGrow: 1,
      padding: theme.spacing(2),
    }),
    body: css({
      label: 'body',
      flexGrow: 1,
      display: 'flex',
      gap: '8px',
      marginBottom: theme.spacing(2),
    }),
  };
}

export default DashboardReportPage;
