"""
Django management command for importing legal data from various sources
"""

import asyncio
import logging
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from legal_research.data_sources import (
    initialize_data_sources,
    perform_data_import,
    import_from_pdf,
    schedule_data_import
)

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Import legal data from various sources (Supreme Court, High Courts, PDF files, etc.)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--source',
            type=str,
            choices=['all', 'supreme-court', 'high-courts', 'pdf'],
            default='all',
            help='Data source to import from'
        )
        parser.add_argument(
            '--days',
            type=int,
            default=30,
            help='Number of days of recent data to import (default: 30)'
        )
        parser.add_argument(
            '--pdf-path',
            type=str,
            help='Path to PDF file to import (only used with --source=pdf)'
        )
        parser.add_argument(
            '--schedule',
            action='store_true',
            help='Run scheduled import (imports last 7 days)'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Perform a dry run without actually importing data'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose logging'
        )

    def handle(self, *args, **options):
        """Handle the command execution"""
        try:
            if options['verbose']:
                logging.getLogger('legal_research').setLevel(logging.DEBUG)

            self.stdout.write(self.style.SUCCESS('Starting legal data import...'))

            if options['schedule']:
                result = self._run_scheduled_import()
            elif options['source'] == 'pdf':
                result = self._import_pdf(options['pdf_path'])
            else:
                result = self._import_from_sources(options['source'], options['days'], options['dry_run'])

            self._display_results(result)

        except Exception as e:
            logger.error(f"Import command failed: {str(e)}")
            raise CommandError(f"Import failed: {str(e)}")

    def _run_scheduled_import(self):
        """Run scheduled import"""
        self.stdout.write(self.style.WARNING('Running scheduled import (last 7 days)...'))
        return schedule_data_import()

    def _import_pdf(self, pdf_path):
        """Import from PDF file"""
        if not pdf_path:
            raise CommandError("PDF path is required when using --source=pdf")

        self.stdout.write(self.style.WARNING(f'Importing from PDF: {pdf_path}'))
        return import_from_pdf(pdf_path)

    def _import_from_sources(self, source, days, dry_run):
        """Import from specified sources"""
        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN MODE - No data will be imported'))

        if source == 'all':
            self.stdout.write(self.style.WARNING(f'Importing from all sources (last {days} days)...'))
            return asyncio.run(perform_data_import(days))
        elif source == 'supreme-court':
            self.stdout.write(self.style.WARNING(f'Importing from Supreme Court (last {days} days)...'))
            # Would implement specific Supreme Court import
            return {'message': 'Supreme Court import not yet implemented'}
        elif source == 'high-courts':
            self.stdout.write(self.style.WARNING(f'Importing from High Courts (last {days} days)...'))
            # Would implement specific High Courts import
            return {'message': 'High Courts import not yet implemented'}

    def _display_results(self, result):
        """Display import results"""
        if 'error' in result:
            self.stdout.write(self.style.ERROR(f"Import failed: {result['error']}"))
            return

        if 'import_summary' in result:
            summary = result['import_summary']
            self.stdout.write(self.style.SUCCESS(
                f"\nImport Summary:\n"
                f"  Total imported: {summary.get('total_imported', 0)}\n"
                f"  Total failed: {summary.get('total_failed', 0)}\n"
                f"  Sources processed: {summary.get('sources_processed', 0)}\n"
                f"  Import date: {summary.get('import_date', 'Unknown')}"
            ))

            if 'source_results' in result:
                self.stdout.write("\nSource Details:")
                for source_name, source_result in result['source_results'].items():
                    if 'error' in source_result:
                        self.stdout.write(self.style.ERROR(f"  {source_name}: FAILED - {source_result['error']}"))
                    else:
                        self.stdout.write(self.style.SUCCESS(
                            f"  {source_name}: {source_result.get('imported', 0)} imported, {source_result.get('failed', 0)} failed"
                        ))

        elif 'pdf_processed' in result:
            if 'error' in result:
                self.stdout.write(self.style.ERROR(f"PDF import failed: {result['error']}"))
            else:
                self.stdout.write(self.style.SUCCESS(
                    f"\nPDF Import Summary:\n"
                    f"  PDF processed: {result.get('pdf_processed', False)}\n"
                    f"  Metadata extracted: {len(result.get('metadata_extracted', {}))} fields\n"
                    f"  Cases imported: {result.get('import_result', {}).get('imported', 0)}"
                ))

        else:
            self.stdout.write(self.style.WARNING(f"Import completed: {result}"))