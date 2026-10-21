"""Safe, repeatable schema upgrades for the application database."""

from sqlalchemy import inspect


def _sqlite_literal(value):
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _column_definition(column, engine):
    definition = column.type.compile(dialect=engine.dialect)
    default = None
    if column.server_default is not None:
        default = column.server_default.arg
    elif column.default is not None and not callable(column.default.arg):
        default = column.default.arg
    if default is not None:
        if hasattr(default, "text"):
            default = default.text
        else:
            default = _sqlite_literal(default)
        definition += f" DEFAULT {default}"
    if not column.nullable and default is None:
        return None
    return definition


def upgrade_database_schema(engine, metadata):
    """Add safe missing columns from SQLAlchemy metadata without changing rows."""
    inspector = inspect(engine)
    added_columns = []

    with engine.begin() as connection:
        for table in metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            existing_columns = {
                column["name"] for column in inspector.get_columns(table.name)
            }
            for column in table.columns:
                if column.name in existing_columns:
                    continue
                definition = _column_definition(column, engine)
                if definition is None:
                    continue
                connection.exec_driver_sql(
                    f"ALTER TABLE {table.name} ADD COLUMN {column.name} {definition}"
                )
                added_columns.append(f"{table.name}.{column.name}")

    return added_columns


def upgrade_medical_document_schema(engine):
    """Backward-compatible wrapper for the medical-document-only upgrade."""
    from models import MedicalDocument

    return upgrade_database_schema(engine, MedicalDocument.metadata)