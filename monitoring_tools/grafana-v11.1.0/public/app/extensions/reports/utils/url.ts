import { Report, StepKey } from '../../types';
import { BASE_URL } from '../constants';
import { initialState } from '../state/reducers';

import { collectVariables } from './variables';

export const getUrlValues = () => {
  if (!window.location.search) {
    return null;
  }

  const urlParams = new URLSearchParams(window.location.search);
  return {
    timeRange: {
      to: urlParams.get('to') || '',
      from: urlParams.get('from') || '',
    },
    dashboard: {
      uid: urlParams.get('db-uid') || '',
      name: urlParams.get('db-name') || '',
    },
    variables: collectVariables(),
  };
};

/**
 * Apply values from URL params as form's default, in case a report is created
 * from dashboard
 * @param report
 */
export const applyUrlValues = (report: Report) => {
  // Do not apply URL values for edited report
  if (report.id) {
    return report;
  }
  const values = getUrlValues();
  if (!values) {
    return report;
  }

  const { timeRange, dashboard, variables } = values;
  let dashboards = [...initialState.report.dashboards];

  if (timeRange?.from && timeRange?.to) {
    dashboards[0] = { ...dashboards[0], timeRange: { ...timeRange, raw: timeRange } };
  }

  if (dashboard.name && dashboard.uid) {
    dashboards[0] = {
      ...dashboards[0],
      dashboard: { uid: dashboard.uid, name: dashboard.name },
    };
  }

  if (variables && Object.keys(variables).length) {
    dashboards[0] = { ...dashboards[0], reportVariables: variables };
  }

  return { ...report, dashboards };
};

export const getTimeRangeFromURL = () => {
  const urlParams = new URLSearchParams(window.location.search);
  const to = urlParams.get('to');
  const from = urlParams.get('from');
  if (from && to) {
    return { from, to };
  }
  return null;
};

export const getSectionUrl = (section: StepKey, id?: string | number) => {
  return `${BASE_URL}/${section}${id ? '/' + id : ''}`;
};

export const getStepUrl = (id?: string | number) => (step: StepKey) => {
  return `${BASE_URL}/${step}${id ? '/' + id : ''}`;
};
