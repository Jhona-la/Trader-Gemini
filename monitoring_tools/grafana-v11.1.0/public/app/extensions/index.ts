import { config, featureEnabled, reportExperimentView } from '@grafana/runtime';
import { contextSrv } from 'app/core/core';
import { ROUTES as CONNECTION_ROUTES } from 'app/features/connections/constants';
import { extraRoutes } from 'app/routes/routes';
import { addExtraMiddleware, addRootReducer } from 'app/store/configureStore';
import { AccessControlAction, DashboardRoutes } from 'app/types';

import { initEnterpriseAdmin } from './admin';
import { initAuthConfig } from './auth-config';
import { SetupSAMLPage } from './auth-config/SAML/SetupSAMLPage';
import { authConfigSAMLReducer } from './auth-config/SAML/state/reducers';
import { initPageBanners } from './banners';
import { DataSourceCachePage } from './caching/DataSourceCachePage';
import { DataSourceCacheUpgradePage } from './caching/DataSourceCacheUpgradePage';
import dataSourceCacheReducers from './caching/state/reducers';
import {
  DataSourcePermissionsPage as ConnectionsDataSourcePermissionsPage,
  DataSourcePermissionsUpgradePage as ConnectionsDataSourcePermissionsUpgradePage,
  DataSourceInsightsUpgradePage as ConnectionsDataSourceInsightsUpgradePage,
  DataSourceInsightsPage as ConnectionsDataSourceInsightsPage,
  DataSourceCachePage as ConnectionsDataSourceCachePage,
  DataSourceCacheUpgradePage as ConnectionsDataSourceCacheUpgradePage,
} from './connections';
import { initLicenseWarnings } from './licensing';
import LicensePage from './licensing/LicensePage';
import { initMetaAnalytics } from './meta-analytics';
import { DataSourceInsightsPage } from './meta-analytics/DataSourceInsights/DataSourceInsightsPage';
import { DataSourceInsightsUpgradePage } from './meta-analytics/DataSourceInsights/DataSourceInsightsUpgradePage';
import metaAnalyticsReducers from './meta-analytics/state/reducers';
import { initNotificationBanners } from './notification-banner';
import { BannerSettingsPage } from './notification-banner/BannerSettingsPage';
import { notificationBannerAPI } from './notification-banner/api';
import { DataSourcePermissionsPage } from './permissions/DataSourcePermissionsPage';
import DatasourcePermissionsUpgradePage from './permissions/UpgradePage';
import { ConfirmAccessPage } from './publicdashboards/ConfirmAccessPage';
import RequestViewAccessPage from './publicdashboards/RequestViewAccessPage';
import { initRecordedQueries } from './recorded-queries';
import { RecordedQueriesConfig } from './recorded-queries/RecordedQueriesConfig';
import { WriteTargetConfig } from './recorded-queries/WriteTargetConfig';
import { recordedQueryReducer } from './recorded-queries/state/reducers';
import { initReporting } from './reports';
import CSVExportPage from './reports/CSVExportPage';
import ReportPage from './reports/ReportForm/ReportPage';
import ReportsList from './reports/ReportsListPage';
import { ReportsSettingsPage as ReportsSettings } from './reports/ReportsSettingsPage';
import ReportsUpgrade from './reports/ReportsUpgradePage';
import { BASE_URL as REPORTS_BASE } from './reports/constants';
import DashboardReportPage from './reports/dashboard/DashboardReportPage';
import reportsReducers from './reports/state/reducers';
import { teamLBACReducer } from './teamLBAC/state/reducers';
import { AccessControlAction as EnterpriseAccessControlAction } from './types';
import { buildExperimentID, ExperimentGroup } from './utils/featureHighlights';
import { initWhitelabeling } from './whitelabeling';

export function addExtensionReducers() {
  if (featureEnabled('caching')) {
    addRootReducer(dataSourceCacheReducers);
  }

  if (featureEnabled('reports')) {
    addRootReducer(reportsReducers);
  }

  if (featureEnabled('analytics')) {
    addRootReducer(metaAnalyticsReducers);
  }

  if (featureEnabled('recordedqueries')) {
    addRootReducer(recordedQueryReducer);
  }

  if (featureEnabled('saml')) {
    addRootReducer(authConfigSAMLReducer);
  }

  if (config.featureToggles.teamHttpHeaders) {
    addRootReducer(teamLBACReducer);
  }

  if (config.featureToggles.notificationBanner) {
    if (featureEnabled('notificationBanner')) {
      addRootReducer({ [notificationBannerAPI.reducerPath]: notificationBannerAPI.reducer });
      addExtraMiddleware(notificationBannerAPI.middleware);
    }
  }
}

function initEnterprise() {
  const highlightsEnabled = config.featureToggles.featureHighlights;
  initLicenseWarnings();
  initReporting();
  initMetaAnalytics();

  if (featureEnabled('saml')) {
    initAuthConfig();
  }

  if (featureEnabled('whitelabeling') || featureEnabled('grafanacloud')) {
    initWhitelabeling();
  }

  if (featureEnabled('recordedqueries')) {
    // TODO: Remove this check when we separate the logic between loading Grafana services and the Public dashboard view
    // this method is called when loading the Grafana services and failing with 401 because it's an auth endpoint,
    // we don't need to load it for the public dashboard view
    if (config.publicDashboardAccessToken === '') {
      initRecordedQueries();
    }
  }

  if (featureEnabled('admin')) {
    initEnterpriseAdmin();
  }

  if (featureEnabled('publicDashboardsEmailSharing')) {
    extraRoutes.push(
      {
        path: '/public-dashboards/:accessToken/request-access',
        component: RequestViewAccessPage,
        chromeless: true,
      },
      {
        path: '/public-dashboards/:accessToken/confirm-access',
        component: ConfirmAccessPage,
        chromeless: true,
      }
    );
  }

  // SAML configuration UI
  if (featureEnabled('saml')) {
    extraRoutes.push({
      path: '/admin/authentication/saml/:step?',
      component: SetupSAMLPage,
    });
  }

  // DataSources / Caching
  const cachePath = '/datasources/edit/:uid/cache';
  const connectionsCachePath = `${CONNECTION_ROUTES.DataSourcesEdit}/cache`;
  if (featureEnabled('caching')) {
    extraRoutes.push({
      path: cachePath,
      component: DataSourceCachePage,
    });
    extraRoutes.push({
      path: connectionsCachePath,
      component: ConnectionsDataSourceCachePage,
    });
  } else if (highlightsEnabled) {
    extraRoutes.push({
      path: `${cachePath}/upgrade`,
      component: DataSourceCacheUpgradePage,
    });
    extraRoutes.push({
      path: `${connectionsCachePath}/upgrade`,
      component: ConnectionsDataSourceCacheUpgradePage,
    });
  }

  // DataSources / Insights
  const insightsPath = '/datasources/edit/:uid/insights';
  const connectionsInsightsPath = `${CONNECTION_ROUTES.DataSourcesEdit}/insights`;
  if (config.analytics?.enabled) {
    if (featureEnabled('analytics')) {
      extraRoutes.push({
        path: insightsPath,
        component: DataSourceInsightsPage,
      });
      extraRoutes.push({
        path: connectionsInsightsPath,
        component: ConnectionsDataSourceInsightsPage,
      });
    } else if (highlightsEnabled) {
      extraRoutes.push({
        path: `${insightsPath}/upgrade`,
        component: DataSourceInsightsUpgradePage,
      });
      extraRoutes.push({
        path: `${connectionsInsightsPath}/upgrade`,
        component: ConnectionsDataSourceInsightsUpgradePage,
      });
    }
  }

  // DataSources / Permissions
  const permissionsPath = '/datasources/edit/:uid/permissions';
  const connectionsPermissionsPath = `${CONNECTION_ROUTES.DataSourcesEdit}/permissions`;
  if (featureEnabled('dspermissions.enforcement')) {
    extraRoutes.push({
      path: permissionsPath,
      component: DataSourcePermissionsPage,
    });
    extraRoutes.push({
      path: connectionsPermissionsPath,
      component: ConnectionsDataSourcePermissionsPage,
    });
  } else if (highlightsEnabled) {
    extraRoutes.push({
      path: permissionsPath + '/upgrade',
      component: DatasourcePermissionsUpgradePage,
    });
    extraRoutes.push({
      path: `${connectionsPermissionsPath}/upgrade`,
      component: ConnectionsDataSourcePermissionsUpgradePage,
    });
  }

  if (config.reporting?.enabled) {
    if (featureEnabled('reports')) {
      extraRoutes.push(
        {
          path: REPORTS_BASE,
          component: ReportsList,
        },
        {
          path: `${REPORTS_BASE}/settings`,
          component: ReportsSettings,
        },
        {
          path: '/d-csv/:uid',
          pageClass: 'dashboard-solo',
          routeName: DashboardRoutes.Normal,
          component: CSVExportPage,
        }
      );

      if (config.featureToggles.newPDFRendering) {
        extraRoutes.push({
          path: '/d-report/:uid/:slug?',
          component: DashboardReportPage,
        });
      }
    } else if (highlightsEnabled) {
      extraRoutes.push({
        path: REPORTS_BASE,
        component: ReportsUpgrade,
      });
    }

    if (featureEnabled('reports.creation')) {
      extraRoutes.push({
        path: `${REPORTS_BASE}/new`,
        component: ReportPage,
      });

      extraRoutes.push({
        path: `${REPORTS_BASE}/:step?/:id?`,
        component: ReportPage,
      });
    }
  }

  const showRecordQuery = featureEnabled('recordedqueries') && config?.recordedQueries?.enabled;
  if (contextSrv.isEditor && showRecordQuery) {
    extraRoutes.push(
      ...[
        {
          path: '/recorded-queries',
          component: RecordedQueriesConfig,
        },
        {
          path: '/recorded-queries/write-target',
          component: WriteTargetConfig,
        },
      ]
    );
  }

  // Notification banner
  if (config.featureToggles.notificationBanner) {
    if (featureEnabled('notificationBanner')) {
      extraRoutes.push({
        path: '/admin/banner-settings',
        component: BannerSettingsPage,
        roles: () => contextSrv.evaluatePermission([EnterpriseAccessControlAction.ActionBannersWrite]),
      });

      initNotificationBanners();
    }
  }
}

// initUnlicensed initialized features which are defined in Enterprise but
// should be available when running without a license.
function initUnlicensed() {
  initPageBanners();

  extraRoutes.push({
    path: '/admin/licensing',
    roles: () =>
      contextSrv.evaluatePermission([
        EnterpriseAccessControlAction.LicensingRead,
        AccessControlAction.ActionServerStatsRead,
      ]),
    component: LicensePage,
  });

  // Report experimentation views
  if (contextSrv.isSignedIn && config.licenseInfo.stateInfo !== 'Licensed') {
    reportExperimentView(
      buildExperimentID(),
      config.featureToggles.featureHighlights ? ExperimentGroup.Test : ExperimentGroup.Control,
      ''
    );
  }
}

export function init() {
  initUnlicensed();
  initEnterprise();
}
