import { featureEnabled } from '@grafana/runtime';
import { ProBadge } from 'app/core/components/Upgrade/ProBadge';
import { config } from 'app/core/config';
import { contextSrv } from 'app/core/services/context_srv';
import { highlightTrial } from 'app/features/admin/utils';
import { addDashboardShareTab, ShareModalTabModel } from 'app/features/dashboard/components/ShareModal';
import { shareDashboardType } from 'app/features/dashboard/components/ShareModal/utils';
import { addDashboardShareTab as addSceneDashboardShareTab } from 'app/features/dashboard-scene/sharing/ShareModal';

import { AccessControlAction, StepKey } from '../types';
import { buildExperimentID } from '../utils/featureHighlights';

import { CreateReportTab } from './CreateReportTab';
import Confirm from './ReportForm/Confirm';
import FormatReport from './ReportForm/FormatReport';
import Schedule from './ReportForm/Schedule';
import SelectDashboard from './ReportForm/SelectDashboard';
import Share from './ReportForm/Share';
import { SharePDF } from './SharePDF';
import { CreateReportTab as CreateReportTabScene } from './dashboard-scene/CreateReportTab';
import { SharePDFTab } from './dashboard-scene/SharePDFTab';

const highlightsEnabled = config.featureToggles.featureHighlights;
const sharePDFTab: ShareModalTabModel = {
  label: 'PDF',
  value: shareDashboardType.pdf,
  component: SharePDF,
};

const createReportTab: ShareModalTabModel = {
  label: 'Report',
  value: shareDashboardType.report,
  tabSuffix:
    (highlightsEnabled && !featureEnabled('reports.creation')) || highlightTrial()
      ? () => ProBadge({ experimentId: buildExperimentID('reporting-tab-badge') })
      : undefined,
  component: CreateReportTab,
};

export function initReporting() {
  if (!config.reporting?.enabled) {
    return;
  }

  if (featureEnabled('reports.creation')) {
    addDashboardShareTab(sharePDFTab);
    addSceneDashboardShareTab(SharePDFTab);

    if (contextSrv.hasPermission(AccessControlAction.ReportingCreate)) {
      addDashboardShareTab(createReportTab);
      addSceneDashboardShareTab(CreateReportTabScene);
    }
  } else if (highlightsEnabled) {
    addDashboardShareTab(createReportTab);
    addSceneDashboardShareTab(CreateReportTabScene);
  }
}

export const reportSteps = [
  { id: StepKey.SelectDashboard, name: 'Select dashboard', component: SelectDashboard },
  { id: StepKey.FormatReport, name: 'Format report', component: FormatReport },
  { id: StepKey.Schedule, name: 'Schedule', component: Schedule },
  { id: StepKey.Share, name: 'Share', component: Share },
  { id: StepKey.Confirm, name: 'Confirm', component: Confirm },
];
