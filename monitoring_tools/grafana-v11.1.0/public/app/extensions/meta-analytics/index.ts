import { config, featureEnabled, registerEchoBackend } from '@grafana/runtime';
import { getConfig } from 'app/core/config';
import { contextSrv } from 'app/core/services/context_srv';

import { initAnalyticsDrawer } from './AnalyticsDrawer/AnalyticsDashNav';
import { MetaAnalyticsBackend } from './MetaAnalyticsBackend';
import { initPresenceIndicators } from './PresenceIndicators';

export const initMetaAnalytics = () => {
  if (!getConfig().analytics?.enabled) {
    return
  }

  if (featureEnabled('analytics')) {
    registerEchoBackend(new MetaAnalyticsBackend({ url: getMetaAnalyticsUrl() }));

    const user = contextSrv.user;
    if (user.isSignedIn) {
      initPresenceIndicators();
      initAnalyticsDrawer();
    }
  } else if (config.featureToggles.featureHighlights) {
    initAnalyticsDrawer();
  }
};

const getMetaAnalyticsUrl = () => {
const publicDashboardAccessToken = getConfig().publicDashboardAccessToken;
  return publicDashboardAccessToken !== ''
    ? `api/public/ma/events/${publicDashboardAccessToken}`
    : 'api/ma/events';
};
