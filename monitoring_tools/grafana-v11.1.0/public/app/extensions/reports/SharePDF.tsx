import { css } from '@emotion/css';
import React, { useMemo, useState } from 'react';

import { config, featureEnabled, reportInteraction } from '@grafana/runtime';
import { Button, Field, LinkButton, Modal, ModalTabContent, RadioButtonGroup, Slider } from '@grafana/ui';
import { ShareModalTabProps } from 'app/features/dashboard/components/ShareModal';
import { ALL_VARIABLE_VALUE } from 'app/features/variables/constants';
import { getVariablesByKey } from 'app/features/variables/state/selectors';

import { ReportLayout, reportLayouts, ReportOrientation, reportOrientations } from '../types';

import { AllTemplateAlert } from './AllTemplateAlert';
import { NoRendererInfoBox } from './RenderingWarnings';
import { UnavailableFeatureInfoBox } from './UnavailableFeatureInfoBox';
import { defaultScaleFactor, reportScales } from './constants';
import { getVariablesUsedInRepeatingPanels } from './state/actions';
import { buildPdfLink } from './utils/pdf';
import { getSelectedVariable, toReportVariables } from './utils/variables';

export const SharePDFBase = ({
  displayQueryVariablesAlert,
  onDismiss,
  isSnapshot,
  variables,
  dashboardUid,
}: {
  displayQueryVariablesAlert?: boolean;
  onDismiss?(): void;
  isSnapshot: boolean;
  variables: Record<string, string[]>;
  dashboardUid: string;
}) => {
  const [orientation, setOrientation] = useState<ReportOrientation>('landscape');
  const [layout, setLayout] = useState<ReportLayout>('grid');
  const [scaleFactor, setScaleFactor] = useState(defaultScaleFactor);

  const pdfUrl = buildPdfLink(orientation, layout, scaleFactor, dashboardUid, variables);

  function onClickSave() {
    reportInteraction('dashboards_sharing_pdf_save_clicked', {
      orientation: orientation,
      layout: layout,
      scaleFactor: scaleFactor,
    });
  }

  if (!config.rendererAvailable) {
    return <NoRendererInfoBox />;
  }

  if (!featureEnabled('reports.pdf')) {
    return (
      <UnavailableFeatureInfoBox
        message="Rendering a dashboard as a PDF document is not available with your current license.
            To enable this feature, update your license."
      />
    );
  }

  if (isSnapshot) {
    return <p className="share-modal-info-text">Sharing a PDF from a snapshot is not supported.</p>;
  }

  return (
    <ModalTabContent icon="file-alt">
      {displayQueryVariablesAlert && <AllTemplateAlert />}
      <p className="share-modal-info-text">Save the dashboard as a PDF document.</p>
      <Field label="Orientation">
        <RadioButtonGroup options={reportOrientations} value={orientation} onChange={setOrientation} />
      </Field>
      <Field label="Layout">
        <RadioButtonGroup options={reportLayouts} value={layout} onChange={setLayout} />
      </Field>
      <Field
        label={'Zoom'}
        description={'Increase the number of rows and columns in table panels.'}
        className={css`
          max-width: 350px;
          // Hide Slider input
          input {
            display: none;
          }
        `}
      >
        <Slider
          value={scaleFactor}
          min={1}
          max={3}
          step={1}
          marks={reportScales}
          onChange={setScaleFactor}
          included={false}
        />
      </Field>
      <Modal.ButtonRow>
        <Button variant="secondary" onClick={onDismiss}>
          Cancel
        </Button>
        <LinkButton variant="primary" href={pdfUrl} target="_blank" rel="noreferrer noopener" onClick={onClickSave}>
          Save as PDF
        </LinkButton>
      </Modal.ButtonRow>
    </ModalTabContent>
  );
};

export const SharePDF = ({ dashboard, onDismiss }: ShareModalTabProps) => {
  const displayQueryVariablesAlert = useMemo(() => {
    const variablesUsedInRepeatingPanelsSet = getVariablesUsedInRepeatingPanels(dashboard.panels);
    const variables = getVariablesByKey(dashboard.uid);
    const selectedVariable = getSelectedVariable(ALL_VARIABLE_VALUE, 'query', variables);
    return selectedVariable && variablesUsedInRepeatingPanelsSet.has(selectedVariable.name);
  }, [dashboard]);

  const variables = useMemo(() => {
    const variables = getVariablesByKey(dashboard.uid);
    return toReportVariables(variables);
  }, [dashboard.uid]);

  return (
    <SharePDFBase
      displayQueryVariablesAlert={displayQueryVariablesAlert}
      isSnapshot={dashboard.isSnapshot()}
      onDismiss={onDismiss}
      variables={variables}
      dashboardUid={dashboard.uid}
    />
  );
};
