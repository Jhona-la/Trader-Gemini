import React, { useCallback, useEffect } from 'react';
import { connect, ConnectedProps } from 'react-redux';

import { GrafanaRouteComponentProps } from 'app/core/navigation/types';
import { Loader } from 'app/features/plugins/admin/components/Loader';

import { EnterpriseStoreState, StepKey } from '../../types';
import { reportSteps } from '../index';
import { loadReport } from '../state/actions';

import { ReportPageContainer } from './ReportPageContainer';

interface OwnProps extends GrafanaRouteComponentProps<{ id: string; step: StepKey }> {}

const mapStateToProps = (state: EnterpriseStoreState, props: OwnProps) => {
  const { report, isLoading } = state.reports;
  const { id: reportId, step: activeStep } = props.match.params;
  return {
    activeStep,
    existingReport: !!report.id,
    isLoading: Boolean(reportId && isLoading),
    report,
    reportId,
  };
};

const mapActionsToProps = {
  loadReport,
};

const connector = connect(mapStateToProps, mapActionsToProps);
export type Props = ConnectedProps<typeof connector> & OwnProps;

export const ReportPage = ({
  existingReport,
  isLoading,
  loadReport,
  report,
  reportId,
  activeStep = StepKey.SelectDashboard,
}: Props) => {
  useEffect(() => {
    if (reportId) {
      loadReport(parseInt(reportId, 10));
    }
  }, [reportId, loadReport]);

  const renderStep = useCallback(() => {
    const Component = reportSteps.find(({ id }) => id === activeStep)?.component;
    if (!Component) {
      return null;
    }
    return <Component reportId={reportId} />;
  }, [activeStep, reportId]);

  return (
    <ReportPageContainer isLoading={isLoading} editMode={existingReport} report={report}>
      {isLoading ? <Loader /> : renderStep()}
    </ReportPageContainer>
  );
};

export default connector(ReportPage);
