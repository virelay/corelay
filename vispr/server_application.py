"""Bokeh plotting handler."""

import json
import random
import sqlite3
import logging
from argparse import Namespace

import h5py
import numpy as np
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, CDSView, IndexFilter, DataTable, TableColumn, Range1d
from bokeh.transform import linear_cmap
from bokeh.plotting import figure
from bokeh.models.widgets import Select, Slider, Button, TextInput
from bokeh.palettes import d3  # pylint: disable=no-name-in-module

from .data import OrigImage, AttrImage


class ServerApplication:
    """Represents the Bokeh server application that renders the data on the client and handles user input."""

    def __init__(self, input_path, attribution_path, analysis_path, wnid_path, wordmap_path, database_path):
        """
        Initializes a new ServerApplication instance.

        Parameters:
        -----------
            input_path: str
                The path to the directory that contains the images of the dataset.
            attribution_path: str
                The path to the HDF5 file that contains the attributions.
            analysis_path: str
                The path to the HDF5 file that contains the analysis.
            wnids_path: str
                The path to the text file that contains the WordNet IDs of the classes of the classifier.
            wordmap_path: str
                The path to the JSON file that contains the mappings between the WordNet IDs and human-readable class
                names.
            database_path: str
                The path to the SQLite3 database that is to be used to store interesting results. If the file already
                exists, then it is used, otherwise a new database is created. When no path is specified, then the save
                feature is disabled.
        """

        # Stores the arguments for later use
        self.input_path = input_path
        self.attribution_path = attribution_path
        self.analysis_path = analysis_path
        self.wordmap_path = wordmap_path
        self.wnid_path = wnid_path

        # Initializes other class members
        self.save_selected_samples_button = None
        self.save_selected_samples_note_text_input = None
        self.sample_source = None
        self.image_source = None
        self.eigen_value_source = None
        self.sample_table = None
        self.interesting_results_table = None
        self.category_select = None
        self.cluster_select = None
        self.original_image_renderer = None
        self.attribution_image_renderer = None
        self.eigen_value_renderer = None

        # Some configuration constants for the plotting
        self.number_of_images = 16
        self.width = 224
        self.height = 224
        self.maximum_width = 4 * self.width
        self.maximum_height = 4 * self.height
        self.colormap = d3['Category20'][20]

        # Gets the logger for the module
        self.logger = logging.getLogger(__name__)

        # Loads the analysis file and extracts the category names
        with h5py.File(self.analysis_path, 'r') as analysis_file:
            self.categories = list(analysis_file)

        # Loads the WordNet IDs file, which contains a mapping from the label numbers to the WNIDs
        with open(self.wnid_path, 'r') as wnid_file:
            self.wnids = wnid_file.read().split('\n')[:-1]

        # Loads the wordmap file, which contains the mapping between the WNIDs and their corresponding cleartext
        # descriptions
        with open(self.wordmap_path, 'r') as wordmap_file:
            self.wordmap = json.load(wordmap_file)

        # Loads the input images and the attributed images
        self.original_image_loader = OrigImage(self.input_path)
        self.attribution_image_loader = AttrImage(self.attribution_path)

        # Initializes the source of the samples, images, and eigen values
        self.sample_source = ColumnDataSource({'i': [], 'x': [], 'y': [], 'cluster': [], 'prediction': []})
        self.image_source = ColumnDataSource({'attribution': [], 'original': [], 'x': [], 'y': []})
        self.eigen_value_source = ColumnDataSource({'x': [], 'y': []})
        self.interesting_results_source = ColumnDataSource({
            'sample_indices': [],
            'category': [],
            'cluster': [],
            'note': []
        })

        # Namespace to store data references
        self.data = Namespace()
        self.data.selected = Namespace()
        self.data.selected.category = self.categories[random.randint(0, len(self.categories))]
        self.data.selected.cluster = None
        self.data.selected.visualization = None
        self.data.interesting_results = Namespace()
        self.data.interesting_results.sample_indices = []
        self.data.interesting_results.category = []
        self.data.interesting_results.cluster = []
        self.data.interesting_results.note = []

        # Initializes the SQLite3 database for storing interesting results
        self.database_path = database_path
        self.is_save_selected_samples_enabled = self.database_path is not None
        if self.is_save_selected_samples_enabled:
            self.initialize_database()

    def initialize_database(self):
        """Initializes the database for storing interesting results."""

        # Establishes a connection to the database, if the database does not exist, yet, then it is created
        with sqlite3.connect(self.database_path) as connection:  # pylint: disable=no-member

            # Initializes the schema of the database, if it does not exist, yet
            try:
                cursor = connection.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS interesting_results(
                        id integer PRIMARY KEY,
                        sample_indices text NOT NULL,
                        category text NOT NULL,
                        cluster text NOT NULL,
                        note text
                    )
                """)
            except sqlite3.Error as sql_error:  # pylint: disable=no-member
                self.logger.error('The schema of the database could not be initialized: %s', sql_error.msg)
                self.is_save_selected_samples_enabled = False

            # Loads the interesting results that are already stored in the database
            try:
                cursor = connection.cursor()
                cursor.execute("SELECT sample_indices, category, cluster, note FROM interesting_results")
                interesting_results = cursor.fetchall()
                for interesting_result in interesting_results:
                    sample_indices, category, cluster, note = interesting_result
                    self.data.interesting_results.sample_indices.append(sample_indices)
                    self.data.interesting_results.category.append(category)
                    self.data.interesting_results.cluster.append(cluster)
                    self.data.interesting_results.note.append(note)
                self.interesting_results_source.data.update({
                    'sample_indices': self.data.interesting_results.sample_indices,
                    'category': self.data.interesting_results.category,
                    'cluster': self.data.interesting_results.cluster,
                    'note': self.data.interesting_results.note
                })
            except sqlite3.Error as sql_error:  # pylint: disable=no-member
                self.logger.error('The interesting result data could not be read from the database: %s', sql_error.msg)
                self.is_save_selected_samples_enabled = False

    def get_wnid_description(self, wnid):
        """
        Gets the human-readable cleartext descriptions for the specified WordNet ID.

        Parameters:
        -----------
            wnid: str
                The WordNet ID that is to be mapped.

        Returns:
        --------
            str:
                Returns the cleartext description for the specified WordNet ID. If the WordNet ID is invalid, 'UNKNOWN'
                is returned.
        """

        return self.wordmap.get(wnid, 'UNKNOWN')

    def select_interesting_result(self, new_interesting_result_index):
        """
        Selects the intersting result.

        Parameters:
        -----------
            new_interesting_result_index:
                The index of the interesting result that was selected by the user.
        """

        # Gets the information about the interesting result
        new_interesting_result_index = new_interesting_result_index[0]
        sample_indices = self.data.interesting_results.sample_indices[new_interesting_result_index]
        category = self.data.interesting_results.category[new_interesting_result_index]
        cluster = self.data.interesting_results.cluster[new_interesting_result_index]

        # Selects the interesting result
        self.category_select.value = category
        self.update_category(category)
        self.cluster_select.value = cluster
        self.update_clusters(cluster)
        self.sample_source.selected.indices = [int(index) for index in sample_indices.split(',')]
        self.update_selection(self.sample_source.selected.indices)

    # load data for a new category
    def update_category(self, new_category):
        """
        Selects a new category and loads the associated data for it.

        Parameters:
        -----------
            new_category:
                The new category that is to be selected.
        """

        # Selects the new category and deselects everything selected previously
        self.data.selected.category = new_category
        self.sample_source.selected.indices = []
        if self.is_save_selected_samples_enabled and self.save_selected_samples_button is not None:
            self.save_selected_samples_button.disabled = True

        # Loads the data of the newly selected category
        with h5py.File(self.analysis_path, 'r') as analysis_file:
            analysis_file = analysis_file[self.data.selected.category]
            self.data.cluster = {key: value[:] for key, value in analysis_file['cluster'].items()}
            self.data.visualization = {key: value[:] for key, value in analysis_file['visualization'].items()}
            self.data.eigen_value = analysis_file['eigenvalue'][:]
            self.data.index = analysis_file['index'][:]

        # Resets the selected cluster and visualization if necessary (that is, when they are not in the newly selected
        # category)
        if self.data.selected.cluster not in self.data.cluster:
            self.data.selected.cluster = next(iter(self.data.cluster))
        if self.data.selected.visualization not in self.data.visualization:
            self.data.selected.visualization = next(iter(self.data.visualization))

        # Loads the attribution data for the newly selected category
        with h5py.File(self.attribution_path, 'r') as attribution_file:
            self.data.prediction = attribution_file['prediction'][self.data.index, :]

        # Updates the sample source with new category data
        description = [self.get_wnid_description(self.wnids[label_id]) for label_id in self.data.prediction.argmax(1)]
        visual_x, visual_y = self.data.visualization[self.data.selected.visualization].T
        visual_cluster = self.data.cluster[self.data.selected.cluster]
        self.sample_source.data.update({
            'i': range(len(visual_x)),
            'x': visual_x,
            'y': visual_y,
            'cluster': visual_cluster,
            'prediction': description
        })

        # Decides which images to show
        indices = list(range(self.number_of_images))
        self.image_source.data.update({
            'attribution': self.attribution_image_loader[self.data.index[indices]],
            'original': list(self.original_image_loader[self.data.index[indices]]),
            'x': (np.arange(self.number_of_images, dtype=int) * self.width) % self.maximum_width,
            'y': (np.arange(self.number_of_images, dtype=int) * self.width) // self.maximum_width * self.height,
        })

        # Updates the eigen value plot
        self.eigen_value_source.data.update({
            'x': range(len(self.data.eigen_value)),
            'y': self.data.eigen_value[::-1],
        })

    def update_selection(self, new_sample_indices):
        """
        Updates the selected sample.

        Parameters:
        -----------
            new_sample_indices: list
                The index of the sample that is to be selected.
        """

        self.sample_table.view = CDSView(source=self.sample_source, filters=[IndexFilter(new_sample_indices)])

        if self.sample_source.selected.indices:  # pylint: disable=no-member
            indices = self.sample_source.selected.indices[:self.number_of_images]  # pylint: disable=no-member
        else:
            indices = list(range(self.number_of_images))
        indices = sorted(indices)

        if indices and self.is_save_selected_samples_enabled:
            self.save_selected_samples_button.disabled = False

        self.image_source.data.update({
            'attribution': self.attribution_image_loader[self.data.index[indices]],
            'original': list(self.original_image_loader[self.data.index[indices]]),
            'x': (np.arange(len(indices), dtype=int) * self.width) % self.maximum_width,
            'y': (np.arange(len(indices), dtype=int) * self.width) // self.maximum_width * self.height,
        })

    def update_clusters(self, new_cluster):
        """
        Updates the clusters.

        Parameters:
        -----------
            new_cluster
                The new cluster.
        """

        self.data.selected.cluster = new_cluster
        self.sample_source.data.update({'cluster': self.data.cluster[self.data.selected.cluster]})

    def update_alpha(self, new_alpha):
        """
        Updates the alpha channel of the original and the attribution images. This slowly hides or reveals the original
        image, which is rendered under the attribution image.

        Parameters:
        -----------
            new_alpha: float
                The new value for the alpha channel.
        """

        self.original_image_renderer.glyph.global_alpha = 1.0 - new_alpha
        self.attribution_image_renderer.glyph.global_alpha = new_alpha

    def save_selected_samples(self):
        """Saves the currently selected samples for later reference."""

        # Checks if the saving of interesting findings is enabled, if not, nothing needs to be done
        if not self.is_save_selected_samples_enabled:
            return

        # Retrieves the information that is to be saved
        # pylint: disable=no-member
        sample_indices = ','.join([str(index) for index in self.sample_source.selected.indices])
        category = self.data.selected.category
        cluster = self.data.selected.cluster
        note = self.save_selected_samples_note_text_input.value

        # Adds the intersting result to the interesting results table
        self.data.interesting_results.sample_indices.append(sample_indices)
        self.data.interesting_results.category.append(category)
        self.data.interesting_results.cluster.append(cluster)
        self.data.interesting_results.note.append(note)
        self.interesting_results_source.data.update({
            'sample_indices': self.data.interesting_results.sample_indices,
            'category': self.data.interesting_results.category,
            'cluster': self.data.interesting_results.cluster,
            'note': self.data.interesting_results.note
        })

        # Saves the data to the database
        with sqlite3.connect(self.database_path) as connection:  # pylint: disable=no-member
            cursor = connection.cursor()
            cursor.execute(
                "INSERT INTO interesting_results(sample_indices, category, cluster, note) VALUES(?, ?, ?, ?)",
                (sample_indices, category, cluster, note)
            )
        self.logger.info('Saved interesting result in the database.')

    def setup_up_bokeh_document(self, document):
        """
        Sets up the Bokeh server document.

        Parameters:
        -----------
            document:
                The document that represents the website displayed by the Bokeh server.
        """

        # Starts the initialization of the document
        self.logger.info('Setting up document...')

        # Loads the data for the initial selected category
        self.update_category(self.data.selected.category)

        # Creates a figure for the eigen values
        eigen_value_figure = figure(
            tools=[],
            plot_width=200,
            plot_height=800,
            min_border=1,
            min_border_left=1,
            toolbar_location="above",
            title="Eigen values"
        )
        self.eigen_value_renderer = eigen_value_figure.scatter(
            'x',
            'y',
            source=self.eigen_value_source,
            size=4,
            color="#3A5785"
        )

        # Creates a figure for the visualization (e.g. TSNE)
        visualization_figure = figure(
            tools="pan,wheel_zoom,box_select,lasso_select,reset",
            plot_width=600,
            plot_height=800,
            min_border=1,
            min_border_left=1,
            toolbar_location="above",
            x_axis_location=None,
            y_axis_location=None,
            title="Visualization",
            active_drag='box_select'
        )
        visualization_colormap = linear_cmap('cluster', self.colormap, 0, len(self.colormap) - 1)
        visualization_figure.scatter(
            'x',
            'y',
            source=self.sample_source,
            size=6,
            color=visualization_colormap,
            nonselection_alpha=0.2,
            nonselection_color=visualization_colormap
        )

        # Creates the table for the selected samples
        sample_columns = [
            TableColumn(field='cluster', title='cluster', width=50),
            TableColumn(field='prediction', title='prediction')
        ]
        self.sample_table = DataTable(source=self.sample_source, columns=sample_columns, width=250, height=800)

        # Creates the table for the interesting results that are stored in the database
        interesting_result_columns = [
            TableColumn(field='sample_indices', title='sample indices'),
            TableColumn(field='category', title='category', width=50),
            TableColumn(field='cluster', title='cluster', width=50),
            TableColumn(field='note', title='note')
        ]
        self.interesting_results_table = DataTable(
            source=self.interesting_results_source,
            columns=interesting_result_columns,
            width=1800,
            height=250
        )

        # Creates a figure containing the original images and attributions
        image_figure = figure(
            tools=[],
            plot_width=800,
            plot_height=800,
            min_border=10,
            min_border_left=10,
            toolbar_location=None,
            x_axis_location=None,
            y_axis_location=None,
            title="Images",
            x_range=Range1d(start=0, end=self.maximum_width),
            y_range=Range1d(start=0, end=self.maximum_height)
        )
        self.attribution_image_renderer = image_figure.image(
            'attribution',
            x='x',
            y='y',
            dw=self.width,
            dh=self.height,
            source=self.image_source,
            palette='Magma256',
            global_alpha=0.0
        )
        self.original_image_renderer = image_figure.image_rgba(
            'original',
            x='x',
            y='y',
            dw=self.width,
            dh=self.height,
            source=self.image_source,
            global_alpha=1.0
        )

        # Adds some widgets to the document for selecting categories, selecting clusters, changing the alpha channel
        # of the images (the attribution images are rendered above the respective original image, changing the alpha
        # reveals the original image or hides it behind the attribution image), and saving the currently selected
        # elements as interesting results
        self.category_select = Select(
            value=self.data.selected.category,
            options=[(category, '{0} ({1})'.format(self.wordmap[category], category)) for category in self.categories]
        )
        self.cluster_select = Select(
            value=self.data.selected.cluster,
            options=list(self.data.cluster),
            width=200
        )
        alpha_slider = Slider(start=0.0, end=1.0, step=0.05, value=0.0, width=400, align='center')
        if self.is_save_selected_samples_enabled:
            self.save_selected_samples_button = Button(label="Save", disabled=True, width=80)
            self.save_selected_samples_note_text_input = TextInput(placeholder='Describe your interesting finding...')

        # Registers the event handlers for the UI controls
        self.category_select.on_change('value', lambda _, __, new_category: self.update_category(new_category))
        self.cluster_select.on_change('value', lambda _, __, new_cluster: self.update_clusters(new_cluster))
        alpha_slider.on_change('value', lambda _, __, new_alpha: self.update_alpha(new_alpha))
        self.sample_source.selected.on_change(  # pylint: disable=no-member
            'indices',
            lambda _, __, new_sample_index: self.update_selection(new_sample_index)
        )
        if self.is_save_selected_samples_enabled:
            self.save_selected_samples_button.on_click(lambda _: self.save_selected_samples())
        self.interesting_results_source.selected.on_change(  # pylint: disable=no-member
            'indices',
            lambda _, __, new_interesting_result_index: self.select_interesting_result(new_interesting_result_index)
        )

        # Sets up the layouting of the document
        if self.is_save_selected_samples_enabled:
            top = row(
                self.category_select,
                self.cluster_select,
                self.save_selected_samples_button,
                self.save_selected_samples_note_text_input
            )
        else:
            top = row(
                self.category_select,
                self.cluster_select
            )
        middle = row(eigen_value_figure, self.sample_table, visualization_figure, column(image_figure, alpha_slider))
        bottom = row(self.interesting_results_table)
        layout = column(top, middle, bottom)
        document.add_root(layout)
        document.title = 'SPRINCL'

        # Finishes the setup of the Bokeh application
        self.logger.info('Ready for service.')
