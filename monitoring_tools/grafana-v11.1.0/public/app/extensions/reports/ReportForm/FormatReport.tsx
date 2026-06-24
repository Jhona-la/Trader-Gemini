import { css, cx } from '@emotion/css';
import React, { useMemo } from 'react';
import { useForm, Controller } from 'react-hook-form';
import { connect, ConnectedProps } from 'react-redux';

import { AppEvents, GrafanaTheme2, urlUtil } from '@grafana/data';
import { featureEnabled, reportInteraction } from '@grafana/runtime';
import { Checkbox, Field, FieldSet, LinkButton, RadioButtonGroup, Slider, useStyles2, Stack } from '@grafana/ui';
import config from 'app/core/config';
import { appEvents } from 'app/core/core';
import { ALL_VARIABLE_VALUE } from 'app/features/variables/constants';
import { getVariablesByKey } from 'app/features/variables/state/selectors';

import {
  EnterpriseStoreState,
  ReportFormat,
  ReportFormData,
  ReportLayout,
  reportLayouts,
  reportOrientations,
  StepKey,
} from '../../types';
import { getRange } from '../../utils/time';
import { AllTemplateAlert } from '../AllTemplateAlert';
import { defaultScaleFactor, reportScales } from '../constants';
import { defaultTimeRange, updateReportProp } from '../state/reducers';
import { dashboardsInvalid } from '../utils/dashboards';
import { canEditReport } from '../utils/permissions';
import { getRendererMajorVersion } from '../utils/renderer';
import { isSelectedVariableInRepeatingPanels } from '../utils/variables';

import ReportForm from './ReportForm';

type FormatData = Pick<ReportFormData, 'formats' | 'options' | 'scaleFactor'>;

const mapStateToProps = (state: EnterpriseStoreState) => {
  const { report } = state.reports;
  return {
    report,
  };
};

const mapActionsToProps = {
  updateReportProp,
};

const connector = connect(mapStateToProps, mapActionsToProps);
export type Props = ConnectedProps<typeof connector> & { reportId?: string };

const descriptions = new Map<ReportLayout, string>([
  ['grid', 'Display the panels in their positions on the dashboard.'],
  ['simple', 'Display one panel per row.'],
]);

interface FormatOptions {
  description: string;
  dependsOn?: ReportFormat;
  hide?: boolean;
}

const formatMap = new Map<ReportFormat, FormatOptions>([
  [ReportFormat.PDF, { description: 'Attach the report as a PDF' }],
  [
    ReportFormat.PDFTablesAppendix,
    {
      description: 'Include table data as PDF appendix',
      dependsOn: ReportFormat.PDF,
      hide: !config.featureToggles.pdfTables,
    },
  ],
  [ReportFormat.Image, { description: 'Embed a dashboard image in the email' }],
  [ReportFormat.CSV, { description: 'Attach a CSV file of table panel data' }],
  [
    ReportFormat.PDFTables,
    { description: 'Attach a separate PDF of table data', hide: !config.featureToggles.pdfTables },
  ],
]);

export const FormatReport = ({ report, updateReportProp, reportId }: Props) => {
  const styles = useStyles2(getStyles);

  const { dashboards, formats, options, name, scaleFactor = defaultScaleFactor } = report || {};
  const {
    handleSubmit,
    control,
    register,
    watch,
    getValues,
    formState: { isDirty },
  } = useForm({
    defaultValues: {
      formats,
      scaleFactor,
      options: { layout: options.layout, orientation: options.orientation, timeRange: options.timeRange },
    },
  });
  const watchLayout = watch('options.layout');
  const watchOrientation = watch('options.orientation');
  const watchFormats = watch('formats');
  const watchScaleFactor = watch('scaleFactor');
  const rendererMajorVersion = getRendererMajorVersion();
  const previewEnabled = featureEnabled('reports.email') && !dashboardsInvalid(report.dashboards);

  const displayQueryVariablesAlert = useMemo(() => {
    const variables = report.dashboards.flatMap((d) => (d.dashboard?.uid ? getVariablesByKey(d.dashboard.uid) : []));
    const isAllSelectedAndUsedInRepeat = isSelectedVariableInRepeatingPanels(ALL_VARIABLE_VALUE, 'query', variables);
    return isAllSelectedAndUsedInRepeat && watchFormats.includes(ReportFormat.PDF);
  }, [report.dashboards, watchFormats]);

  const saveData = ({ formats, options, scaleFactor }: FormatData) => {
    if (isDirty) {
      updateReportProp({ ...report, options: { ...report.options, ...options }, formats, scaleFactor });
    }
  };

  const getFormData = () => {
    const { formats, options, scaleFactor } = getValues();

    return { options: { ...report.options, ...options }, formats, scaleFactor };
  };

  const getPreviewPDFUrl = () => {
    if (dashboardsInvalid(dashboards)) {
      return undefined;
    }

    const params: any = {
      title: name,
      scaleFactor: watchScaleFactor,
    };

    if (watchOrientation) {
      params.orientation = watchOrientation;
    }

    if (watchLayout) {
      params.layout = watchLayout;
    }

    if (watchFormats && watchFormats.includes(ReportFormat.PDFTablesAppendix)) {
      params.includeTables = 'true';
    }

    params.dashboards = JSON.stringify(
      dashboards.map((db) => {
        const { from, to } = getRange(db.timeRange).raw || defaultTimeRange.raw;
        return {
          dashboard: { uid: db.dashboard?.uid },
          timeRange: { from: from.valueOf().toString(), to: to.valueOf().toString() },
          reportVariables: db.reportVariables,
        };
      })
    );

    return urlUtil.appendQueryToUrl(`api/reports/render/pdfs/`, urlUtil.toUrlParams(params));
  };

  return (
    <ReportForm
      activeStep={StepKey.FormatReport}
      onSubmit={handleSubmit(saveData)}
      confirmRedirect={isDirty}
      getFormData={getFormData}
      reportId={reportId}
      pageActions={[
        <LinkButton
          onClick={() => previewEnabled && reportInteraction('reports_preview_pdf')}
          icon={'external-link-alt'}
          href={getPreviewPDFUrl()}
          size="xs"
          target="_blank"
          rel="noreferrer noopener"
          variant="secondary"
          disabled={!previewEnabled}
          key={'preview'}
        >
          Preview PDF
        </LinkButton>,
      ]}
    >
      <FieldSet label={'2. Format report'} disabled={!canEditReport}>
        {displayQueryVariablesAlert && <AllTemplateAlert />}
        <FieldSet>
          <Stack direction={'column'} alignItems={'start'}>
            {[...formatMap].map(([name, opts]) => {
              // Get onChange from the register, to be able to customise checkbox onChange
              const { onChange: onFormatChange, ...formatFields } = register('formats');
              return (
                !opts.hide &&
                (!opts.dependsOn || watchFormats.includes(opts.dependsOn)) && (
                  <Checkbox
                    {...formatFields}
                    hidden={opts.hide || (opts.dependsOn && watchFormats.includes(opts.dependsOn))}
                    key={name}
                    htmlValue={name}
                    label={opts.description}
                    className={cx(opts.dependsOn && styles.subCheckbox)}
                    onChange={(val) => {
                      if (name === 'csv') {
                        const enabled = val.currentTarget.checked;
                        if (enabled && rendererMajorVersion !== null && rendererMajorVersion < 3) {
                          appEvents.emit(AppEvents.alertError, [
                            'To export CSV files, you must update the Grafana Image Renderer plugin.',
                          ]);
                        }
                      }
                      onFormatChange(val);
                    }}
                  />
                )
              );
            })}
          </Stack>
        </FieldSet>
        {(watchFormats.includes(ReportFormat.PDF) || watchFormats.includes(ReportFormat.PDFTables)) && (
          <FieldSet label={'Style the PDF'}>
            <Field label="Orientation">
              <Controller
                name={'options.orientation'}
                control={control}
                render={({ field: { ref, ...field } }) => {
                  return <RadioButtonGroup {...field} options={reportOrientations} size={'md'} />;
                }}
              />
            </Field>
            <Field label="Layout" description={descriptions.get(watchLayout)}>
              <Controller
                name={'options.layout'}
                control={control}
                render={({ field: { ref, ...field } }) => {
                  return <RadioButtonGroup {...field} options={reportLayouts} size={'md'} />;
                }}
              />
            </Field>
            <Field
              label={'Zoom'}
              description={'Zoom in to enlarge text, or zoom out to see more data (like table columns) per panel.'}
              className={styles.sliderField}
            >
              <Controller
                name={'scaleFactor'}
                control={control}
                render={({ field: { ref, value, ...field } }) => (
                  <Slider
                    {...field}
                    value={Number(value)}
                    min={1}
                    max={3}
                    step={1}
                    marks={reportScales}
                    included={false}
                  />
                )}
              />
            </Field>
          </FieldSet>
        )}
      </FieldSet>
    </ReportForm>
  );
};

const getStyles = (theme: GrafanaTheme2) => {
  return {
    sliderField: css`
      max-width: 350px;
      // Hide Slider input
      input {
        display: none;
      }
    `,
    subCheckbox: css`
      margin-left: ${theme.spacing(2)};
    `,
  };
};

export default connector(FormatReport);
