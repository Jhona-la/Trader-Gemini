import { saveAs } from 'file-saver';
import React, { Component } from 'react';
import { connect, ConnectedProps } from 'react-redux';

import { applyFieldOverrides, PanelData, dateTimeFormat, toCSV, LoadingState, FieldConfigSource } from '@grafana/data';
import { config } from '@grafana/runtime';
import { GrafanaContext, GrafanaContextType } from 'app/core/context/GrafanaContext';
import { GrafanaRouteComponentProps } from 'app/core/navigation/types';
import { SoloPanel } from 'app/features/dashboard/containers/SoloPanelPage';
import { PanelModel, DashboardModel } from 'app/features/dashboard/state';
import { initDashboard } from 'app/features/dashboard/state/initDashboard';
import { StoreState } from 'app/types';

export interface DashboardPageRouteParams {
  uid?: string;
  type?: string;
  slug?: string;
}

export interface OwnProps
  extends GrafanaRouteComponentProps<DashboardPageRouteParams, { panelId?: string; timezone?: string }> {
  dashboard: DashboardModel | null;
}

export interface State {
  panel: PanelModel | null;
  notFound: boolean;
}

const mapStateToProps = (state: StoreState) => ({
  dashboard: state.dashboard.getModel(),
});

const mapDispatchToProps = {
  initDashboard,
};

const connector = connect(mapStateToProps, mapDispatchToProps);
export type Props = ConnectedProps<typeof connector> & OwnProps;

export class CSVExportPage extends Component<Props, State> {
  declare context: GrafanaContextType;
  static contextType = GrafanaContext;

  state: State = {
    panel: null,
    notFound: false,
  };

  componentDidMount() {
    const { match, route } = this.props;

    this.props.initDashboard({
      urlSlug: match.params.slug,
      urlUid: match.params.uid,
      urlType: match.params.type,
      routeName: route.routeName,
      fixUrl: false,
      keybindingSrv: this.context.keybindings,
    });
  }

  getPanelId(): number {
    return parseInt(this.props.queryParams.panelId ?? '0', 10);
  }

  componentDidUpdate(prevProps: Props) {
    const { dashboard } = this.props;

    if (!dashboard) {
      return;
    }

    // we just got a new dashboard
    if (!prevProps.dashboard || prevProps.dashboard.uid !== dashboard.uid) {
      const panel = dashboard.getPanelByUrlId(this.props.queryParams.panelId!);

      if (!panel) {
        this.setState({ notFound: true });
        return;
      }

      this.setState({ panel });

      panel
        .getQueryRunner()
        .getData({ withTransforms: true, withFieldConfig: true })
        .subscribe({
          next: (data) => this.onDataUpdate(data),
        });
    }
  }

  onDataUpdate(panelData: PanelData) {
    const { panel } = this.state;
    if (!panelData || (panelData.state !== LoadingState.Done && panelData.state !== LoadingState.Streaming)) {
      return;
    }

    const { dashboard } = this.props;
    const data = panelData.series;
    // We need to apply field config even though it was already applied in the PanelQueryRunner.
    // That's because transformers create new fields and data frames, so i.e. display processor is no longer there
    const timeZone = this.props.queryParams.timezone || dashboard?.timezone;
    let emptyFieldConfigSource: FieldConfigSource = {
      defaults: {},
      overrides: [],
    };
    const dataFrames = applyFieldOverrides({
      data,
      theme: config.theme2,
      fieldConfig: panel?.fieldConfig || emptyFieldConfigSource,
      replaceVariables: (value: string) => {
        return value;
      },
      timeZone: timeZone,
    });

    const dataFrameCsv = toCSV([dataFrames[0]], {});

    const blob = new Blob([String.fromCharCode(0xfeff), dataFrameCsv], {
      type: 'text/csv;charset=utf-8',
    });
    const fileName = `${panel?.getDisplayTitle()}-data-${dateTimeFormat(new Date())}.csv`;
    saveAs(blob, fileName);
  }

  render() {
    return (
      <SoloPanel
        dashboard={this.props.dashboard}
        notFound={this.state.notFound}
        panel={this.state.panel}
        panelId={this.getPanelId()}
        timezone={this.props.queryParams.timezone}
      />
    );
  }
}

export default connector(CSVExportPage);
