import { isEmpty } from 'lodash';
import React, { useCallback, useEffect } from 'react';
import { connect, ConnectedProps } from 'react-redux';

import { NavModelItem } from '@grafana/data';
import { Page } from 'app/core/components/Page/Page';
import { GrafanaRouteComponentProps } from 'app/core/navigation/types';
import { Loader } from 'app/features/plugins/admin/components/Loader';

import { samlSteps } from '..';
import { EnterpriseStoreState, SAMLStepKey } from '../../types';

import { loadSAMLSettings } from './state/actions';

interface OwnProps extends GrafanaRouteComponentProps<{ step: SAMLStepKey }> {}

function mapStateToProps(state: EnterpriseStoreState, props: OwnProps) {
  const { isLoading, samlSettings } = state.samlConfig;
  const { step: activeStep } = props.match.params;
  return {
    isLoading,
    samlSettings,
    activeStep,
  };
}

const mapDispatchToProps = {
  loadSAMLSettings,
};

const connector = connect(mapStateToProps, mapDispatchToProps);
export type Props = OwnProps & ConnectedProps<typeof connector>;

const pageNav: NavModelItem = {
  text: 'Setup SAML single sign on',
  subTitle: `This guided application will guide you through the configuration
  process of SAML. You will need to follow steps and visit your Identity
  Provider's application to connect it with Grafana (The Service provider).
  This guided application will help you keep track of your progress.`,
  icon: 'shield',
  id: 'SAML',
};

export const SetupSAMLPageUnconnected = ({
  isLoading,
  samlSettings,
  activeStep,
  loadSAMLSettings,
}: Props): JSX.Element => {
  useEffect(() => {
    loadSAMLSettings();
  }, [loadSAMLSettings]);

  const renderStep = useCallback(() => {
    const Component = samlSteps.find(({ id }) => id === activeStep)?.component;
    if (!Component) {
      return null;
    }
    return <Component />;
  }, [activeStep]);

  return (
    <Page navId="authentication" pageNav={pageNav}>
      <Page.Contents>{isLoading || isEmpty(samlSettings) ? <Loader /> : renderStep()}</Page.Contents>
    </Page>
  );
};

export const SetupSAMLPage = connector(SetupSAMLPageUnconnected);
